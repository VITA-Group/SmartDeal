from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .resnet_se_mask import *
from .mobilenetv2_se_mask import *

model_names = [
    'VGG', 'DPN', 'DPN26', 'DPN92', 'LeNet',
    'PreActBlock', 'SENet', 'SENet18',
    'SepConv', 'PNASNet', 'PNASNetA', 'PNASNetB',
    'DenseNet', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161',
    'Inception', 'GoogLeNet',
    'ShuffleBlock', 'ShuffleNet', 'ShuffleNetG2', 'ShuffleNetG3', 'ShuffleNetV2',
    'ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    'ResNeXt', 'ResNeXt29_2x64d', 'ResNeXt29_4x64d', 'ResNeXt29_8x64d', 'ResNeXt29_32x4d',
    'PreActResNet', 'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101', 'PreActResNet152',
    'MobileNet', 'MobileNetV2', 'SEMaskMobileNetV2',
    'EfficientNet', 'EfficientNetB0',
    'SEMaskResNet', 'SEMaskResNet18', 'SEMaskResNet34', 'SEMaskResNet50', 'SEMaskResNet101', 'SEMaskResNet152'
]

