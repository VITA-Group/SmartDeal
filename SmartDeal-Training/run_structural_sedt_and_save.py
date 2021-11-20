import os
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models

from se.alg_structural import smart_state_dict


model_names = models.model_names

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
""" Start """
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet34', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: ResNet34)')
parser.add_argument('--dest-arch', '-da', metavar='DEST-ARCH', default='SEMaskResNet34', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: ResNet34)')
parser.add_argument('--resume', '-r', type=str, help='resume from specified model')
parser.add_argument('--save-dir', type=str, default=None, help='Output dir')
parser.add_argument('--save-name', type=str, default=None, help='Base name of saved files.')
# parser.add_argument('--dest-dir', '-dd', type=str, help='Destination directory.')
""" SED arguments """
parser.add_argument('--iternum', type=int, default=30, help='Maximum number of iterations in SED.')
parser.add_argument('--threshold', type=float, help='Threshold for element wise sparsifying.')
parser.add_argument('--threshold-decay', type=float, default=0.5, help='Decay ratio of threshold in SED')
parser.add_argument('--scale', action='store_true', help='Whether scare rows before Ce quantization.')
parser.add_argument('--tol', type=float, default=1e-10, help='Tolerance to quit the algorithm.')
parser.add_argument('--rcond', type=float, default=1e-10, help='rcond param in NumPy lsq solver.')
parser.add_argument('--threshold-row', action='store_true', help='Whether to sparsify in a row-wise way.')
parser.add_argument('--init-method', type=str, default='trivial', help='Initialization method in SED.')
parser.add_argument('--max-C', type=float, default=32, help='Max value in Ce matrices')
""" End """
args = parser.parse_args()

assert args.resume is not None
if args.save_dir is not None:
    dirname = args.save_dir
    if not os.path.exists(dirname):
        os.makedirs(dirname)
else:
    dirname = os.path.dirname(args.resume)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
print('==> Building model..')
net = models.__dict__[args.arch]()
# net = models.__dict__[args.arch](**qargs)
# net = VGG('VGG19')
# net = ResNet34()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
se_net = models.__dict__[args.dest_arch](threshold=args.threshold)
se_net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    se_net = torch.nn.DataParallel(se_net)
    cudnn.benchmark = True

print('==> Resuming from checkpoint {}..'.format(args.resume))
# assert os.path.isdir(exp_dir), 'Error: no checkpoint directory `{}` found!'.format(exp_dir)
checkpoint = torch.load(args.resume)
state_dict = checkpoint['net']
net.load_state_dict(state_dict, strict=False)
best_acc = checkpoint['acc']
epoch = checkpoint['epoch']

if 'MobileNetV2' in args.arch:
    threshold_list = np.loadtxt('./mobilenetv2-structure-percentage.txt')
elif 'ResNet18' in args.arch:
    threshold_list = np.loadtxt('./resnet18-structure-percentage.txt')
else:
    raise ValueError('Invalid architecture.')

decompose_opts = dict(
    decompose_iternum = args.iternum,
    decompose_threshold = args.threshold,
    decompose_decay = args.threshold_decay,
    decompose_scale = args.scale,
    decompose_tol = args.tol,
    decompose_rcond = args.rcond,
    threshold_row = args.threshold_row,
    init_method = args.init_method,
    structural_pruning=True,
    pruning_type='population',
    threshold_list=threshold_list,
    max_C = args.max_C,
)
""" Perfrom SED """
decomps = smart_state_dict(state_dict, **decompose_opts)
net.load_state_dict(state_dict, strict=False)

""" Collect all Ce and B matrices. """
clt_tensor_bs = dict()
clt_tensor_ces = dict()
for l in range(len(decomps.keys())):
    layer = decomps['l%d'%(l+1)]
    clt_b = []
    clt_ce = []
    if layer['type'] == 'conv':
        prefix = 'k'
    else:
        prefix = 'r'
    for k in range(len(layer.keys())-2):
        current = layer[(prefix+'%d')%(k+1)]
        clt_b.append(np.stack(current['Bs'], axis=0))
        clt_ce.append(np.stack(current['Ces'], axis=0))
    clt_tensor_bs['l%d'%(l+1)] = np.concatenate(clt_b, axis=0)
    clt_tensor_ces['l%d'%(l+1)] = np.concatenate(clt_ce, axis=0)

""" Assert that the collections above are correct. """
i = 0
for k,v in state_dict.items():
    if ('weight' in k) and (len(v.shape) >= 2):
        i = i + 1
        layer = decomps['l%d'%i]
        if len(layer.keys()) <= 2:
            continue

        if v.data.is_cuda:
            w = v.data.detach().cpu().numpy()
        else:
            w = v.data.detach().numpy()

        ce = clt_tensor_ces['l%d'%i]
        b = clt_tensor_bs['l%d'%i]
        wr = np.einsum('bij,bjk->bik', ce, b)
        if layer['type'] == 'conv':
            if w.shape[-1] == 1:
                wr = wr.reshape(w.shape[0], -1, 1, 1)[:,0:w.shape[1],:,:]
            else:
                try:
                    wr = wr.reshape(*w.shape)
                except ValueError:
                    print(layer['type'], layer['shape'])
                    print(w.shape)
        else:
            wr = wr.reshape(w.shape[0], -1)[:,0:w.shape[1]]
        try:
            assert np.allclose(wr, w)
        except AssertionError as e:
            print('***** w *****')
            print(w.shape)
            print(w[-1,-1,:,:])
            print('***** wr *****')
            print(wr[-1,-1,:,:])
            print('***** ce *****')
            print(ce.shape)
            print(ce[-1,-3:,:])
            print('***** b *****')
            print(b.shape)
            print(b[-1])
            # print(wr.shape)
            # print(wr, w)
            # raise e

""" Assign parameters in the SE-decomposed model. """
i = 0
for m, sem in zip(net.modules(), se_net.modules()):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        i += 1
        if isinstance(sem, (torch.nn.Conv2d, torch.nn.Linear)):
            sem.weight = m.weight
            sem.bias = m.bias
            continue
        sem.B.data = torch.from_numpy(clt_tensor_bs['l%d'%i]).float().to(device)
        sem.C.data = torch.from_numpy(clt_tensor_ces['l%d'%i]).float().to(device)
        sem.bias = m.bias
        weight = sem.get_weight(mask=False)
        try:
            assert torch.allclose(weight.float(), m.weight)
        except AssertionError as e:
            err = (weight.float() - m.weight).abs().max()
            if err > 0.1:
                print(err)
                # raise e
    if isinstance(m, torch.nn.BatchNorm2d):
        sem.weight = m.weight
        sem.bias = m.bias
        sem.running_mean = m.running_mean
        sem.running_var = m.running_var
        sem.num_batches_tracked = m.num_batches_tracked

if args.save_name is not None:
    base_name = args.save_name.split('.')[0]
    save_name = base_name + '.pth.tar'
    decomps_name = base_name.replace('ckpt', 'decomps') + '.npz'
else:
    save_name = 'ckpt-{}-structural-dj.pth.tar'.format(args.threshold)
    decomps_name = 'decomps-{}-structural-dj.pth.tar'.format(args.threshold)
save_path = os.path.join(dirname, save_name)
decomps_path = os.path.join(dirname, decomps_name)
torch.save({
    'net': se_net.state_dict(),
    'acc': best_acc,
    'epoch': epoch,
}, save_path)
np.savez(decomps_path, **decomps)
print('Decomposed joint model is saved into {}'.format(save_path))

