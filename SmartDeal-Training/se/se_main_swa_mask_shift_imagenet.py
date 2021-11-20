import argparse
import os
import random
import shutil
import time
import warnings
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as
import se_resnet_mask
import swa_utils
from se_conv_mask import SEConv2d
from se_linear_mask import SELinear
from quantize import sparsify_and_nearestpow2

from tensorboardX import SummaryWriter

model_names = sorted(name for name in se_resnet_mask.__dict__
    if name.islower() and not name.startswith("__")
    and callable(se_resnet_mask.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='imagenet', type=str, metavar='D',
                    help='a brief description of the dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# threshold argument
parser.add_argument('--threshold', type=float, default=4e-3, metavar='TH',
                    help='threshold that promotes sparsity in C (default: 4e-3)')
parser.add_argument('--min_C', type=float, default=0.00390625, metavar='mC',
                    help='minimal magnitude in Ce matrices. choose between \{0.00390625, 0.0078125\}.')
parser.add_argument('--max_C', type=float, default=None, metavar='MC',
                    help='maximal magnitude in Ce matrices. not set by default')
# swa arguments
parser.add_argument('--swa_start', type=float, default=161, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR',
                    help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

best_acc1 = 0
best_acc5 = 0
best_swa_acc1 = 0
best_swa_acc5 = 0
swa_n = 0
writer = None

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_swa_acc1
    global best_acc5, best_swa_acc5
    global swa_n
    global writer
    args.gpu = gpu

    exp_desc = ("{arch}+mask+swa+shift-{dataset}-lr={lr}-threshold={threshold:2.1e}-"
                "swa_start={swa_start}-swa_lr={swa_lr}-swa_c={swa_c_epochs}".format(
                    arch=args.arch, dataset=args.dataset,
                    batch_size=args.batch_size, lr=args.lr, threshold=args.threshold,
                    swa_start=args.swa_start, swa_lr=args.swa_lr, swa_c_epochs=args.swa_c_epochs))
    exp_dir = os.path.join('logs', exp_desc)
    print("working directory:\n{exp_dir}".format(exp_dir=exp_dir))
    # check_arguments(args)
    writer = SummaryWriter(exp_dir)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, threshold=args.threshold)
        # swa model
        swa_model = models.__dict__[args.arch](pretrained=True, threshold=args.threshold)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = se_resnet_mask.__dict__[args.arch](threshold=args.threshold)
        # swa model
        swa_model = se_resnet_mask.__dict__[args.arch](threshold=args.threshold)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            swa_model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            swa_model = torch.nn.parallel.DistributedDataParallel(swa_model, device_ids=[args.gpu])
        else:
            model.cuda()
            swa_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            swa_model = torch.nn.parallel.DistributedDataParallel(swa_model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        swa_model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            # swa model
            raise NotImplementedError('The SWA versions AlexNet and VGG have not '
                                      'been implemented yet')
        else:
            model = torch.nn.DataParallel(model).cuda()
            swa_model = torch.nn.DataParallel(swa_model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'epoch' in checkpoint.keys():
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                if checkpoint['optimizer'] is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                model.load_state_dict(checkpoint)
                print("=> loaded checkpoint '{}'".format(args.resume))
            # swa parts in checkpoint
            if ('swa_state_dict' in checkpoint.keys() and
                checkpoint['swa_state_dict'] is not None):
                swa_model.load_state_dict(checkpoint['swa_state_dict'])
            if 'swa_n' in checkpoint.keys() and checkpoint['swa_n'] is not None:
                swa_n = checkpoint['swa_n']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for m, swa_m in zip(model.modules(), swa_model.modules()):
        if isinstance(m, (SEConv2d, SELinear,)):
            m.set_mask()
            swa_m.mask.data = m.mask.data.clone()
    print('All masks are set....')

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        writer.add_scalar('accuracies/acc1', acc1, epoch+1)
        writer.add_scalar('accuracies/acc5', acc5, epoch+1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # swa part
        if ((epoch + 1) >= args.swa_start and
            (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):
            swa_utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            swa_utils.bn_update(train_loader, swa_model)
            swa_acc1, swa_acc5 = validate(val_loader, swa_model, criterion, args)
            writer.add_scalar('accuracies/swa_acc1', swa_acc1, epoch+1)
            writer.add_scalar('accuracies/swa_acc5', swa_acc5, epoch+1)


        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'swa_state_dict': swa_model.state_dict(),
                'swa_n': swa_n,
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_dir=exp_dir, filename="checkpoint_{}.pth.tar".format(epoch+1))


def train(train_loader, model, criterion, optimizer, epoch, args):
    global swa_n

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # update Ce matrices using shifting operations
        for m in model.modules():
            if isinstance(m, (SEConv2d, SELinear)):
                grad_C = m.C.grad
                m.C.grad = None
                grad_C_sign = grad_C.sign().float()
                # Ce non-zero and gradient non-zero
                update_C_pow = grad_C_sign * m.C.sign().float()
                update_C_mul = 2 ** update_C_pow
                # Ce zero (not in the mask) and gradient non-zero
                update_C_add = (m.C == 0.0).float() * m.C.mask * grad_C_sign * args.min_C
                # update C
                new_C = m.C.data * update_C_mul + update_C_add
                if args.max_C is not None:
                    new_C.clamp_(-args.max_C, args.max_C)
                m.C.data = sparsify_and_nearestpow2(new_C, args.min_C)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, save_dir='.', filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


# def check_arguments(args):
#     key_args = OrderedDict(
#         arch='Network Architecture',
#         dataset='Dataset',
#     )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    t = epoch / args.swa_start
    lr_ratio = args.swa_lr / args.lr
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    lr = args.lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()