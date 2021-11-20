'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from math import log2, ceil

import models
from utils import progress_bar, setup_logger
from utils_data import CIFAR10
from utils_swa import moving_average, bn_update
from utils_optim import SGD
from utils_quantize import sparsify_and_nearestpow2
from tensorboardX import SummaryWriter

model_names = models.model_names

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet34', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: ResNet34)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int, help='Training batch size.')
parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs.')
parser.add_argument('--lr-decay', default=0.1, type=float, help='Learning rate decay rate.')
parser.add_argument('--milestones', type=str, default='80,120',
                    help='Milestones for learning rate and dC_threshold decays.')
parser.add_argument('--resume', '-r', type=str, help='resume from specified model')
parser.add_argument('--swa-resume', type=str, default=None,
                    help='Resume SWA model from a saved torch model.')
parser.add_argument('--task', type=str, default='adapt',
                    help='Task type that decides dataset splitting method.')
parser.add_argument('--seed', metavar='SEED', default=0, type=int,
                    help='Random seed for dataset splitting.')
parser.add_argument('--source', action='store_true', help='Pretrain models on the source dataset.')
parser.add_argument('--target', action='store_true', help='Train models on the target dataset.')
parser.add_argument('--exp-dir', type=str, default=None,
                    help='Checkpoints and logs will be saved to results/`exp_dir` folder.')
parser.add_argument('--exp-comment', type=str, default=None, help='Comment appended to ``exp_dir``.')
parser.add_argument('--validate', action='store_true', help='Validation mode.')
parser.add_argument('--verbose', action='store_true', help='Print informations to debug.')
parser.add_argument('--writer', action='store_true', help='Use writer to record interested values.')
parser.add_argument('--writer-comment', type=str, default='', help='Comment for TensorboardX writer.')
parser.add_argument('--writer-layer', type=str, default='module.conv1', help='Interested layer for debugging.')
parser.add_argument('--print-dC', action='store_true', help='Print dC.')
parser.add_argument('--print-layer', type=str, default='module.conv1', help='Interested layer for debugging.')
parser.add_argument('--linear-only', action='store_true', help='Fix layers other than linear layers.')
parser.add_argument('--linear-only-end-epoch', type=int, default=-1, help='Ending epoch to train linear layers only.')
parser.add_argument('--return-linear-input', action='store_true', help='Return inputs to linear layers too.')
parser.add_argument('--ignore-C', action='store_true', help='Do not update Ce.')
parser.add_argument('--bn-update-first', action='store_true', help='Update bn first before first epoch.')
""" SE model arguments """
parser.add_argument('--threshold', type=float, default=4e-3,
                    help='Threshold in SED-T.')
parser.add_argument('--quant-each-step', action='store_true',
                    help='Sparsify and quantize coefficient matrcies after each training step.')
""" Bucket Switch arguments """
parser.add_argument('--switch', action='store_true',
                    help='Use Bucket Switch update scheme.')
parser.add_argument('--switch-bar', type=int, default=1,
                    help='Minimal times of accumulated gradient directions before an update.')
parser.add_argument('--max-C', type=float, default=None, metavar='MC',
                    help='maximal magnitude in Ce matrices. not set by default')
parser.add_argument('--dC-threshold', type=float, default=-1,
                    help='Threshold that filter small changes in Ce')
parser.add_argument('--dC-threshold-decay', type=float, default=0.1, help='``dC_threshold`` decay rate.')
""" SWA arguments """
parser.add_argument('--swa', action='store_true', help='Use SWA training technique.')
parser.add_argument('--swa-start', type=float, default=161, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swa-lr', type=float, default=0.05, metavar='LR',
                    help='SWA LR (default: 0.05)')
parser.add_argument('--swa-c-epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_swa_acc = 0  # best test swa accuracy
swa_n = 0
writer = None
batch_counter = 0
dC_threshold = args.dC_threshold

assert args.source != args.target

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.task == 'adapt':
    np.random.seed(args.seed)
    plabels = np.random.permutation(10).tolist()
    used_samples = None
    used_labels = plabels[:5] if args.source else plabels[5:]
elif args.task == 'finetune':
    np.random.seed(args.seed)
    pidx = np.random.permutation(50000).tolist()
    used_samples = pidx[:25000] if args.source else pidx[25000:]
    used_labels = None
elif args.task == 'full':
    used_samples = None
    used_labels = None
else:
    raise ValueError('Invalid task type.')

trainset = CIFAR10(root='./data', used_labels=used_labels, used_samples=used_samples,
                   train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', used_labels=used_labels, used_samples=None,
                  train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# classes = tuple([classes[i] for i in used_labels])

# Model
kwargs = {}
if 'SEMask' in args.arch:
    kwargs['threshold'] = args.threshold
print('==> Building model..')
net = models.__dict__[args.arch](**kwargs)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

""" Create SWA model if needed. """
if args.swa:
    swa_net = models.__dict__[args.arch](**kwargs)
    swa_net = swa_net.to(device)
    if device == 'cuda':
        swa_net = torch.nn.DataParallel(swa_net)

""" Switch arguments setting. """
if args.switch:
    for i in range(-10, 1):
        if 2**i >= args.threshold:
            args.min_C = 2**i
            break

if args.exp_dir is None:
    arch = args.arch
    arch += '-qes' if args.quant_each_step else ''
    arch += '-swa-lr{}'.format(args.swa_lr) if args.swa else ''
    arch += '-switch-bar{}-maxC{}'.format(args.switch_bar, args.max_C) if args.switch else ''
    if args.resume is not None:
        arch += '-structural' if 'structural' in args.resume and args.target else ''
    arch += ('-dC-th{:.0e}-decay{}'.format(args.dC_threshold, args.dC_threshold_decay)
             if args.dC_threshold > 0.0 else '')
    exp_dir = os.path.join(
        'results',
        '{}_{}_{}_{}'.format(arch, args.seed, args.task, 'source' if args.source else 'target')
    )
else:
    exp_dir = os.path.join('results', args.exp_dir)
if args.exp_comment is not None:
    exp_dir += args.exp_comment

print('exp dir is {}'.format(exp_dir))

if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

if args.writer:
    writer = SummaryWriter(comment=args.writer_comment)

log_file = 'log-test.txt' if args.validate else 'log.txt'
log_path = os.path.join(exp_dir, log_file)
logger = setup_logger(log_path)

if args.resume is not None:
    # Load checkpoint.
    logger('==> Resuming from checkpoint {}..'.format(args.resume))
    # assert os.path.isdir(exp_dir), 'Error: no checkpoint directory `{}` found!'.format(exp_dir)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    if args.swa:
        if args.swa_resume is not None:
            swa_checkpoint = torch.load(args.swa_resume)
            swa_net.load_state_dict(swa_checkpoint['net'])
        else:
            swa_net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('best_acc: {}'.format(best_acc))
    print('best epoch: {}'.format(start_epoch))

if args.target and not args.validate:
    if args.bn_update_first:
        print('Update bn layers first before first epoch...')
        source_labels = plabels[:5] if used_labels is not None else None
        source_samples = pidx[:25000] if used_samples is not None else None
        assert source_labels is not None or source_samples is not None
        source_trainset = CIFAR10(root='./data', used_labels=source_labels, used_samples=source_samples,
                           train=True, download=True, transform=transform_train)
        source_trainloader = torch.utils.data.DataLoader(source_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        bn_update(source_trainloader, net)
        print('bn update done')
    if args.task == 'adapt':
        net.module.linear.reset_parameters()
        if args.swa:
            swa_net.module.linear.reset_parameters()
        logger('linear layers reset.')
    best_acc = 0.0
    start_epoch = 0

if 'SEMask' in args.arch:
    for m in net.modules():
        if hasattr(m, 'mask'):
            m.set_mask()
    if args.swa:
        for m in swa_net.modules():
            if hasattr(m, 'mask'):
                m.set_mask()

milestones = [int(stone) for stone in args.milestones.split(',')]
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
parameters = net.module.linear.parameters() if args.linear_only else net.parameters()
optimizer = SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
if args.linear_only and args.linear_only_end_epoch >= 0:
    optimizer_all = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)
if args.linear_only and args.linear_only_end_epoch >= 0:
    scheduler_all = optim.lr_scheduler.MultiStepLR(optimizer_all, milestones=milestones, gamma=args.lr_decay)


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    # return args.lr * factor, args.dC_threshold * factor
    return args.lr * factor, args.dC_threshold


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# Training
def train(epoch):
    global batch_counter, dC_threshold
    logger('\nEpoch: %d' % epoch)
    if args.dC_threshold > 0.0:
        logger('\ndC_threshold: %f' % dC_threshold)
    net.train()
    # reset ``dC_counter`` in SE layers
    if args.switch:
        for m in net.modules():
            if hasattr(m, 'mask'):
                m.reset_dC_counter()
    train_loss = 0
    correct = 0
    total = 0
    if args.linear_only:
        if epoch < args.linear_only_end_epoch:
            optim = optimizer
        else:
            optim = optimizer_all
    else:
        optim = optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_counter += 1
        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()

        if 'SEMask' in args.arch and args.quant_each_step:
            for m in net.modules():
                if hasattr(m, 'mask'):
                    with torch.no_grad():
                        m.C_prev = m.sparsify_and_quantize_C()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if args.print_dC:
            for name, m in net.named_modules():
                if name == args.print_layer:
                    dC = optim.get_d(m.C)
                    print(dC)

        if args.ignore_C:
            for name, m in net.named_modules():
                if not hasattr(m, 'mask'):
                    continue
                with torch.no_grad():
                    m.C.grad = None

        if args.switch:
            # update Ce matrices using ``Bucket Switching`` scheme
            for name, m in net.named_modules():
                if not hasattr(m, 'mask'):
                    continue
                with torch.no_grad():
                    qC = m.sparsify_and_quantize_C()
                    # grad_C = m.C.grad
                    dC = optim.get_d(m.C)
                    if dC is None:
                        continue
                    if args.dC_threshold > 0.0 and dC_threshold > 0.0:
                        dC[dC.abs() <= dC_threshold] = 0.0
                    m.C.grad = None
                    dC_sign = dC.sign().float()
                    # update ``dC_counter``
                    m.dC_counter.add_(dC_sign)
                    activated = m.dC_counter.abs() == args.switch_bar
                    # if activated.any():
                    #     print('Ce is updated!!')
                    dC_sign = m.dC_counter.sign() * activated.float()
                    # Ce non-zero and gradient non-zero
                    dC_pow = dC_sign * qC.sign().float()
                    dC_mul = 2 ** dC_pow
                    # Ce zero (not in the mask) and gradient non-zero
                    dC_add = (qC == 0.0).float() * m.mask * dC_sign * args.min_C
                    # update C
                    new_C = qC.data * dC_mul + dC_add
                    if args.max_C is not None:
                        new_C.clamp_(-args.max_C, args.max_C)
                    m.C.data = new_C
                    # reset activated counters to 0
                    m.dC_counter[activated] = 0.0
                    # m.C.data = sparsify_and_nearestpow2(new_C, args.threshold)

        optim.step()

        if 'SEMask' in args.arch and args.quant_each_step:
            for name, m in net.named_modules():
                if hasattr(m, 'mask'):
                    with torch.no_grad():
                        m.C.data = m.sparsify_and_quantize_C()
                    if args.writer and args.writer_layer is not None and name == args.writer_layer:
                        diff = m.C != m.C_prev
                        diff_ratio = float(diff.sum()) / diff.numel()
                        writer.add_scalar(name+'/diff_ratio', diff_ratio, batch_counter)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, swa=False):
    global best_acc
    global best_swa_acc
    testnet = swa_net if swa else net
    testnet.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = testnet(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    is_best = acc > best_swa_acc if swa else acc > best_acc
    if is_best:
        logger('Saving..')
        state = {
            'net': testnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        save_name = 'ckpt-swa-start{}.pth.tar'.format(args.swa_start) if swa else 'ckpt.pth.tar'
        torch.save(state, os.path.join(exp_dir, save_name))
        if swa:
            best_swa_acc = acc
        else:
            best_acc = acc
    if swa:
        logger('Epoch: [{}] swa acc: {} best swa acc: {}'.format(epoch+1, acc, best_swa_acc))
    else:
        logger('Epoch: [{}] acc: {} best acc: {}'.format(epoch+1, acc, best_acc))

def validate():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.return_linear_input:
                outputs, linear_inputs = net(inputs, True)
                # logger(linear_inputs)
            else:
                outputs = net(inputs)
            # break
            # logger(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    logger('Test acc at epoch {}: {}%'.format(start_epoch, acc))

if args.validate:
    if args.bn_update_first:
        bn_update(trainloader, net)
    validate()
    if 'SEMask' in args.arch:
        i = 0
        sets = []
        numel_C = 0
        numel_nnz_C = 0
        numel_B = 0
        # numels = []
        # masks = []
        # sparsities = []
        logger('\nCheck sparsity mask.')
        logger('id\tnumel\tmask\tsparsity')
        for m in net.modules():
            if hasattr(m, 'mask'):
                numel = m.mask.numel()
                mask = (m.mask == 0.0).sum()
                C = m.sparsify_and_quantize_C()
                numel_C += C.numel()
                numel_nnz_C += (C != 0.0).sum()
                numel_B += m.B.numel()
                sparsity = (C == 0.0).sum()
                logger('{id}\t{numel}\t{mask}\t{sparsity}'.format(
                    id=i+1, numel=numel, mask=mask, sparsity=sparsity))
                vset = set(C.abs().detach().to('cpu').numpy().reshape(-1).tolist())
                sets.append(vset)
                i += 1
        vset = set().union(*sets)
        C_bit = ceil(log2(len(list(vset)) - 1)) + 1
        C_size = float(numel_nnz_C * C_bit) / 8.0 / 1024.0**2
        En_size = float(numel_C) * 1.0 / 8.0 / 1024.0**2
        B_size = numel_B * 32.0 / 8.0 / 1024.0**2
        logger('\nAll (absolute) values in coefficient matrices include')
        logger(sorted(list((vset))))
        logger('model sizes:')
        logger('Ce: {}\tB: {}\tEn: {}\tTotal: {}'.format(C_size, B_size, En_size, C_size+B_size+En_size))
    else:
        i = 0
        sets = []
        logger('\nCheck number of parameters')
        numel = 0
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                numel += m.weight.numel()
        size = numel * 32.0 / 8.0 / 1024.0**2
        logger('total model size is: {} MB'.format(size))

else:
    for epoch in range(start_epoch, start_epoch+args.epochs):
        lr, dC_threshold = schedule(epoch)
        # lr = schedule(epoch)
        adjust_learning_rate(optimizer, lr)
        if args.linear_only and args.linear_only_end_epoch >= 0:
            adjust_learning_rate(optimizer_all, lr)
        train(epoch)
        test(epoch)
        # scheduler.step()
        # if epoch == milestones[-1]:
            # args.dC_threshold *= args.dC_threshold_decay
        # if args.linear_only and args.linear_only_end_epoch >= 0:
            # scheduler_all.step()
        # SWA part
        if args.swa:
            if ((epoch + 1) >= args.swa_start and
                (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):
                moving_average(swa_net, net, 1.0 / (swa_n + 1))
                swa_n += 1
                bn_update(trainloader, swa_net)
                logger('\n********* SWA model testing *********')
                test(epoch, swa=True)
