"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul

from quantize import QConv2d, QLinear

NUM_BITS = 8
NUM_BITS_WEIGHT = 8

__all__ = [
    'QResNet18'
]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = QConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QConv2d(
                    in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                    num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        weight = self.conv1.weight.data
        # # nz = (weight != 0.0).sum().detach().cpu().item()
        # # nz = float(nz) / weight.numel()
        # nz = 1 - 0.821324
        # flops = reduce(mul, out.shape[1:]) * reduce(mul, weight.shape[1:]) * nz * 2.0
        # print(flops)
        out = self.bn2(self.conv2(out))
        weight = self.conv1.weight.data
        # # nz = (weight != 0.0).sum().detach().cpu().item()
        # # nz = float(nz) / weight.numel()
        # nz = 1 - 0.821324
        # flops = reduce(mul, out.shape[1:]) * reduce(mul, weight.shape[1:]) * nz * 2.0
        # print(flops)
        out += self.shortcut(x)
        if len(self.shortcut) > 1:
            weight = self.shortcut[0].weight.data
            # # nz = (weight != 0.0).sum().detach().cpu().item()
            # # nz = float(nz) / weight.numel()
            # nz = 1 - 0.821324
            # flops = reduce(mul, out.shape[1:]) * reduce(mul, weight.shape[1:]) * nz * 2.0
            # print(flops)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = QConv2d(
            in_planes, planes, kernel_size=1, bias=False,
            num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QConv2d(planes, self.expansion*planes, kernel_size=1, bias=False,
            num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QConv2d(
                    in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                    num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # print(x.shape, self.conv1.weight.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape, self.conv2.weight.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        # print(out.shape, self.conv3.weight.shape)
        out = self.bn3(self.conv3(out))
        # print(out.shape, self.shortcut[0].weight.shape if len(self.shortcut) > 1 else None)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class QResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(QResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = QConv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
            num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = QLinear(
            512*block.expansion, num_classes,
            num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        weight = self.conv1.weight.data
        # # nz = (weight != 0.0).sum().detach().cpu().item()
        # # nz = float(nz) / weight.numel()
        # nz = 1 - 0.821324
        # flops = reduce(mul, out.shape[1:]) * reduce(mul, weight.shape[1:]) * nz * 2.0
        # print(flops)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        weight = self.linear.weight.data
        # # nz = (weight != 0.0).sum().detach().cpu().item()
        # # nz = float(nz) / weight.numel()
        # nz = 1 - 0.821324
        # flops = weight.shape[0] * weight.shape[1]**2.0 * nz * 2.0
        # print(flops)
        return out


def QResNet18():
    return QResNet(BasicBlock, [2,2,2,2])

def QResNet34():
    return QResNet(BasicBlock, [3,4,6,3])

def QResNet50():
    return QResNet(Bottleneck, [3,4,6,3])

def QResNet101():
    return QResNet(Bottleneck, [3,4,23,3])

def QResNet152():
    return QResNet(Bottleneck, [3,8,36,3])


def test():
    net = QResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
