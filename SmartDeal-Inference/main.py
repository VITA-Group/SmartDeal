from __future__ import print_function

import argparse
import os
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg
import qresnet

import utils
from smart_exchange import smart_net

from tensorboardX import SummaryWriter

# decompose_opts = dict(decompose_iternum=30,
#                       decompose_threshold=5e-3,
#                       decompose_decay=0.1,
#                       decompose_scale=True,
#                       decompose_tol=1e-10,
#                       decompose_rcond=1e-10,
#                       save_Ce=True,
#                       init_method='trivial')

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))


def str2bool(v):
    return v.lower() in ['yes', '1', 'true', 'y']


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    # choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--quantize', action='store_true',
                    help='quantize the network layers, activations and inputs.')
parser.add_argument('--num_bits', default=None, type=int,
                    help='the number of bits that the input and activations '
                         'will be quantized to.')
parser.add_argument('--num_bits_weight', default=None, type=int,
                    help='the number of bits that the weights will be '
                         'quantized to.')
parser.add_argument('--decompose_iternum', default=30, type=int,
                    help='number of maximum iterations in smart exchange.')
parser.add_argument('--decompose_threshold', default=1e-3, type=float,
                    help='threshold that promotes sparsity in smart exchange.')
parser.add_argument('--decompose_decay', default=0.5, type=float,
                    help='decay rate if the threshold is too large.')
parser.add_argument('--decompose_scale', default=False, type=str2bool,
                    help='scale the Ce matrix for better quantization.')
parser.add_argument('--decompose_tol', default=1e-10, type=float,
                    help='tolerance of stopping condition in smart exchange.')
parser.add_argument('--decompose_rcond', default=1e-10, type=float,
                    help='rcond for the least squares in smart exchange')
parser.add_argument('--init_method', default='trivial', type=str,
                    help='initialization method in smart exchange.'
                         'select between `trivial` | `ksvd` (not implemented)')
parser.add_argument('--threshold_row', default=False, action='store_true',
                    help='threshold by rows before thresholding elementwisely.')
parser.add_argument('--decompose_first', default=False, action='store_true',
                    help='execute smart exchange decomposition before training.')
parser.add_argument('--structural_pruning', default=False, action='store_true',
                    help='flag for structural pruning before SmartExchange.')
parser.add_argument('--prune_type', default='population', type=str,
                    help='type of structural pruning. select among value|population|energy.')
parser.add_argument('--threshold-list-file', default='./resnet18-structure-percentage.txt', type=str,
                    help='threshold list for structral pruning.')
parser.add_argument('--max-C', type=float, default=32, help='Max value in Ce matrices')


best_prec1 = 0
best_prec2 = 0
swa_best_prec2 = 0
writer = None
swa_n = 0

def main():
    global args, best_prec1, best_prec2, swa_best_prec2, writer, swa_n
    args = parser.parse_args()
    writer = SummaryWriter(args.save_dir)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # quantization arguments
    qargs = dict(quantize=args.quantize, num_bits=args.num_bits,
                 num_bits_weight=args.num_bits_weight)

    threshold_list = np.loadtxt(args.threshold_list_file)

    decompose_opts = dict(
        decompose_iternum=args.decompose_iternum,
        decompose_threshold=args.decompose_threshold,
        decompose_decay=args.decompose_decay,
        decompose_scale=args.decompose_scale,
        decompose_tol=args.decompose_tol,
        decompose_rcond=args.decompose_rcond,
        threshold_row=args.threshold_row,
        init_method=args.init_method,
        structural_pruning=args.structural_pruning,
        prune_type=args.prune_type,
        threshold_list=threshold_list,
        max_C = args.max_C,
    )

    model = qresnet.__dict__[args.arch]()
    swa_model = qresnet.__dict__[args.arch]()

    model = torch.nn.DataParallel(model)
    model.cuda()

    swa_model = torch.nn.DataParallel(swa_model)
    swa_model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['net'], strict=False)
            swa_model.load_state_dict(checkpoint['net'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        validate(val_loader, swa_model, criterion)
        return

    # args.start_epoch = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # update swa model after each epoch
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1
        utils.bn_update(train_loader, swa_model)
        utils.quant_buffer_copy(swa_model, model)

        # evaluate on validation set before smart_exchange decomposition
        print('\nvalidation BEFORE smart exchange decomposition\n')
        prec1 = validate(val_loader, model, criterion)
        writer.add_scalar('acc_before_se', prec1, epoch - args.start_epoch + 1)

        # if epoch > 0 and epoch % 5 == 0:
        decomps = smart_net(model, **decompose_opts)
        swa_decomps = smart_net(swa_model, **decompose_opts)
        print('\nvalidation AFTER smart exchange decomposition\n')
        prec2 = validate(val_loader, model, criterion)
        writer.add_scalar('acc_after_se', prec2, epoch - args.start_epoch + 1)
        print('\nSWA model validation AFTER smart exchange decomposition\n')
        swa_prec2 = validate(val_loader, swa_model, criterion)
        writer.add_scalar('swa_acc_after_se', swa_prec2, epoch - args.start_epoch + 1)

        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        is_best = prec2 > best_prec2
        best_prec2 = max(prec2, best_prec2)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'net': model.state_dict(),
                # 'best_prec1': best_prec1,
                'acc': best_prec2,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.tar'))
            np.savez(os.path.join(args.save_dir, 'decompositions.npz'), **decomps)

        is_best = swa_prec2 > swa_best_prec2
        swa_best_prec2 = max(swa_prec2, swa_best_prec2)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'net': swa_model.state_dict(),
                # 'best_prec1': best_prec1,
                'acc': swa_best_prec2,
            }, is_best, filename=os.path.join(args.save_dir, 'swa_checkpoint.tar'.format(epoch+1)))
            np.savez(os.path.join(args.save_dir, 'swa_decompositions.npz'.format(epoch+1)), **swa_decomps)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # exit()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
