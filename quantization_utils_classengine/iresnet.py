import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from quantization_utils_classengine.quant_modules import QuantAct, Quant_Linear, Quant_Conv2d, QuantActPreLu
from collections import OrderedDict
import copy

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200', 'iresnet500']
using_ckpt = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False, use_stn=False,
                 use_sa=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05, )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        #### 512*1*7*7
        #### num_class,识别最后一层的线性层,通常表示每个ID的簇中心;
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)  #### shape为(B,512*expansion*7*7)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)  #### shape为(B,512*expansion*7*7)->(B,512(embedding_length))
        x = self.features(x)
        return x


# def quantize_model(model,weight_bit=None):
#     pass


def quantize_model(model, weight_bit=None, act_bit=None, full_precision_flag=False):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # if not (weight_bit) and not (act_bit ):
    #    weight_bit = self.settings.qw
    #    act_bit = self.settings.qa
    # quantize convolutional and linear layers
    if type(model) == nn.Conv2d:
        quant_mod = Quant_Conv2d(weight_bit=weight_bit, full_precision_flag=full_precision_flag)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=weight_bit, full_precision_flag=full_precision_flag)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.PReLU:
        quant_mod = QuantActPreLu(act_bit=act_bit, full_precision_flag=full_precision_flag)
        quant_mod.set_param(model)
        return quant_mod
    # quantize all the activation
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.PReLU:
        return nn.Sequential(*[model, QuantAct(activation_bit=act_bit, full_precision_flag=full_precision_flag)])
    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential or isinstance(model, nn.Sequential):
        mods = OrderedDict()
        for n, m in model.named_children():
            if isinstance(m, IBasicBlock):
                mods[n] = nn.Sequential(*[
                    quantize_model(m, weight_bit=weight_bit, act_bit=act_bit, full_precision_flag=full_precision_flag),
                    QuantAct(activation_bit=act_bit, full_precision_flag=full_precision_flag)])
            else:
                mods[n] = quantize_model(m, weight_bit=weight_bit, act_bit=act_bit,
                                         full_precision_flag=full_precision_flag)
        return nn.Sequential(mods)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
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


def freeze_model(model):
    """
    freeze the activation range
    即：在量化过程中，激活层（QuantAct）的量化参数（如量化范围）被设置为固定值，之后不会随着进一步的训练或使用而改变
    """
    if type(model) == QuantAct:
        model.fix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)
        return model


def unfreeze_model(model):
    """
    unfreeze the activation range
    即：在量化过程中，取消之前固定的激活范围设置，使得激活层（QuantAct）的量化参数可以根据新的数据再次调整。
    """
    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)
        return model


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)


def iresnet500(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet500', IBasicBlock, [15, 65, 150, 15], pretrained,
                    progress, **kwargs)
