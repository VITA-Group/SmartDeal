'''
 # @ Author: Xiaohan Chen
 # @ Email: chernxh@tamu.edu
 # @ Create Time: 2019-05-26 23:54
 # @ Modified by: Xiaohan Chen
 # @ Modified time: 2019-08-17 21:19
 # @ Description:
 '''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from se import SEConv2d, SELinear

THRESHOLD = 4e-3

__all__ = ['SEMaskResNet', 'SEMaskResNet18', 'SEMaskResNet34', 'SEMaskResNet50', 'SEMaskResNet101', 'SEMaskResNet152']


'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    return SEConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, threshold=THRESHOLD)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return SEConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                    bias=False, threshold=THRESHOLD)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv1 = nn.Conv2d(in_planes, planes, stride=stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion*planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SEMaskResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(SEMaskResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = SELinear(512*block.expansion, num_classes, threshold=THRESHOLD)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def SEMaskResNet18(**kwargs):
    global THRESHOLD
    THRESHOLD = kwargs.pop('threshold', 4e-3)
    print('\nThreshold set to {:2.1e}\n'.format(THRESHOLD))
    return SEMaskResNet(BasicBlock, [2,2,2,2], **kwargs)

def SEMaskResNet34(**kwargs):
    global THRESHOLD
    THRESHOLD = kwargs.pop('threshold', 4e-3)
    print('\nThreshold set to {:2.1e}\n'.format(THRESHOLD))
    return SEMaskResNet(BasicBlock, [3,4,6,3], **kwargs)

def SEMaskResNet50(**kwargs):
    global THRESHOLD
    THRESHOLD = kwargs.pop('threshold', 4e-3)
    print('\nThreshold set to {:2.1e}\n'.format(THRESHOLD))
    return SEMaskResNet(Bottleneck, [3,4,6,3], **kwargs)

def SEMaskResNet101(**kwargs):
    global THRESHOLD
    THRESHOLD = kwargs.pop('threshold', 4e-3)
    print('\nThreshold set to {:2.1e}\n'.format(THRESHOLD))
    return SEMaskResNet(Bottleneck, [3,4,23,3], **kwargs)

def SEMaskResNet152(**kwargs):
    global THRESHOLD
    THRESHOLD = kwargs.pop('threshold', 4e-3)
    print('\nThreshold set to {:2.1e}\n'.format(THRESHOLD))
    return SEMaskResNet(Bottleneck, [3,8,36,3], **kwargs)


def test():
    net = SEMaskResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

if __name__ == "__main__":
    test()

