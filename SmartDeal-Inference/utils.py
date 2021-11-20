import os
import numpy as np
import torch

from quantize import QuantMeasure

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def quant_buffer_copy(swa_model, model):
    for swa_m, m in zip(swa_model.modules(), model.modules()):
        if isinstance(swa_m, QuantMeasure) and isinstance(m, QuantMeasure):
            swa_m.running_zero_point.data = m.running_zero_point.clone()
            swa_m.running_range.data = m.running_range.data.clone()


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def model_info_from_decoms(decomps_file):
    dc = np.load(decomps_file, allow_pickle=True)
    tnumelc = 0
    tnumelb = 0
    tnumzero = 0
    cvset = set()
    for k, v in dc.items():
        v = v.tolist()
        lnumelc = 0
        lnumelb = 0
        lnumzero = 0
        for kk, vv in v.items():
            if not (kk.startswith('r') or kk.startswith('k')):
                continue
            if 'Bs' not in vv.keys():
                continue
            for B in vv['Bs']:
                lnumelb += B.size
            for Ce in vv['Ces']:
                lnumelc += Ce.size
                lnumzero += int(np.sum(Ce == 0.0))
                vset = set(np.abs(Ce.reshape(-1)).tolist())
                cvset = cvset.union(vset)

        tnumelc += lnumelc
        tnumelb += lnumelb
        tnumzero += lnumzero
        lsratio = lnumzero / lnumelc
        print('{layer:5s}\t{numel:10d}\t{numzero:10d}\t{sratio:.6f}'.format(
                    layer=k, numel=lnumelc, numzero=lnumzero, sratio=lsratio))

    # Sparsity
    print('\ntotal\t{:10d}\t{:10d}\t{:.6f}'.format(
        tnumelc, tnumzero, tnumzero / tnumelc))

    # Model size
    from math import ceil, log2
    nbitc = ceil(log2(len(list(cvset)) - 1)) + 1
    print('\nNumber of bits for Ce: {}'.format(nbitc))
    nbitb = 8
    sizec = nbitc * (tnumelc - tnumzero) / 8 / 1024**2
    sizec_enc = 1 * tnumelc / 8 / 1024**2
    sizeb = nbitb * tnumelb / 8 / 1024**2
    print('\nModel size:')
    print('Ce-nz: {}\t Ce-enc: {}\t B: {}\t Total: {}'.format(
        sizec, sizec_enc, sizeb, sizec + sizec_enc + sizeb))

