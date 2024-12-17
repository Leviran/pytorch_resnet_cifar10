'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet18', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

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
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2])


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


import torch.nn as nn
from collections import OrderedDict
import copy
from quantization_utils_classengine.quant_modules import QuantAct, Quant_Linear, Quant_Conv2d, QuantActPreLu


def quantize_model(model, weight_bit=None, act_bit=None, full_precision_flag=False):
    """
    Recursively quantize a pretrained single-precision ResNet model to int8 quantized model.
    model: pretrained single-precision model
    weight_bit: bit width for weights (e.g., 8 for int8)
    act_bit: bit width for activations (e.g., 8 for int8)
    full_precision_flag: whether to use full precision for certain layers
    """

    if isinstance(model, nn.Conv2d):
        # Quantize convolutional layers
        quant_mod = Quant_Conv2d(weight_bit=weight_bit, full_precision_flag=full_precision_flag)
        quant_mod.set_param(model)
        return quant_mod

    elif isinstance(model, nn.Linear):
        # Quantize fully connected (Linear) layers
        quant_mod = Quant_Linear(weight_bit=weight_bit, full_precision_flag=full_precision_flag)
        quant_mod.set_param(model)
        return quant_mod

    elif isinstance(model, nn.PReLU):
        # Quantize PReLU activation layers
        quant_mod = QuantActPreLu(act_bit=act_bit, full_precision_flag=full_precision_flag)
        quant_mod.set_param(model)
        return quant_mod

    elif isinstance(model, nn.ReLU) or isinstance(model, nn.ReLU6) or isinstance(model, nn.PReLU):
        # Quantize other activations (ReLU, ReLU6, etc.)
        return nn.Sequential(
            model,
            QuantAct(activation_bit=act_bit, full_precision_flag=full_precision_flag)
        )

    elif isinstance(model, nn.Sequential):
        # Recursively quantize modules within a Sequential container
        mods = OrderedDict()
        for n, m in model.named_children():
            mods[n] = quantize_model(m, weight_bit=weight_bit, act_bit=act_bit, full_precision_flag=full_precision_flag)
        return nn.Sequential(mods)

    else:
        # Deepcopy the model and recursively quantize its sub-modules
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            # Skip 'norm' layers (e.g., BatchNorm)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(q_model, attr, quantize_model(mod, weight_bit=weight_bit, act_bit=act_bit,
                                                      full_precision_flag=full_precision_flag))
        return q_model

"""
这个函数可以指定不同层使用不同的量化位宽
"""
def quantize_model_mix(model, bit_config, full_precision_flag=False):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model.
    model: pretrained single-precision model.
    bit_config: dict containing quantization config for each layer.
    """

    # Helper function to get bit width from config based on the layer name.
    def get_bit_width(layer_name, bit_config, default_bit=8):
        """
        Get the quantization bit-width for a given layer from the bit_config.
        """
        if layer_name in bit_config:
            return bit_config[layer_name]
        return default_bit  # Default to 8 if not found in config.

    # If the model is a convolutional layer
    if isinstance(model, nn.Conv2d):
        # Get quantization bit-width for weight and activation for Conv2d layers
        layer_name_weight = f'quant_convbn{model.weight.shape[0]}'  # e.g., 'quant_convbn1'
        weight_bit = get_bit_width(layer_name_weight, bit_config, 8)

        layer_name_act = f'quant_act{model.weight.shape[0]}'  # e.g., 'quant_act1'
        act_bit = get_bit_width(layer_name_act, bit_config, 8)

        quant_mod = Quant_Conv2d(weight_bit=weight_bit, full_precision_flag=full_precision_flag)
        quant_mod.set_param(model)
        return quant_mod

    # If the model is a fully connected layer (fc)
    elif isinstance(model, nn.Linear):
        layer_name = 'quant_output'  # For fully connected layers, we'll map to 'quant_output'
        weight_bit = get_bit_width(layer_name, bit_config, 8)
        act_bit = get_bit_width(f'{layer_name}_act', bit_config, 8)

        quant_mod = Quant_Linear(weight_bit=weight_bit, full_precision_flag=full_precision_flag)
        quant_mod.set_param(model)
        return quant_mod

    # If the model is an activation function (ReLU, PReLU, etc.)
    elif isinstance(model, (nn.ReLU, nn.PReLU)):
        layer_name = 'quant_act'  # Map to 'quant_act' for activations
        act_bit = get_bit_width(layer_name, bit_config, 8)
        return nn.Sequential(model, QuantAct(activation_bit=act_bit, full_precision_flag=full_precision_flag))

    # For special activation layers such as PreLU
    elif isinstance(model, nn.PReLU):
        layer_name = 'quant_act'  # Map 'quant_act' to PReLU
        act_bit = get_bit_width(layer_name, bit_config, 8)
        return nn.Sequential(model, QuantActPreLu(activation_bit=act_bit, full_precision_flag=full_precision_flag))

    # Recursively apply quantization to layers in nn.Sequential (e.g., blocks, networks)
    elif isinstance(model, nn.Sequential):
        mods = OrderedDict()
        for n, m in model.named_children():
            mods[n] = quantize_model_mix(m, bit_config, full_precision_flag)
        return nn.Sequential(mods)

    # Handle other layers (e.g., IBasicBlock, etc.)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:  # Skip normalization layers
                setattr(q_model, attr, quantize_model_mix(mod, bit_config, full_precision_flag))
        return q_model

if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()