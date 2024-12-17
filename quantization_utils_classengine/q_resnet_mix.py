import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from quantization_utils_classengine.quant_modules import QuantAct, Quant_Linear, Quant_Conv2d, QuantActPreLu
from collections import OrderedDict
import copy
import logging

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
                 groups=1, base_width=64, dilation=1, stage_num=1, unit_num=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.stage_num = stage_num
        self.unit_num = unit_num

        # 定义原始的卷积和批归一化层
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)

        self.downsample = downsample
        self.stride = stride

        # 定义量化模块，并命名与 bit_config.py 中的键一致
        # 例如，'stage1.unit1.quant_act'
        self.quant_convbn1 = Quant_Conv2d()  # 对应 'stageX.unitY.quant_convbn1'
        self.quant_act = QuantActPreLu()  # 对应 'stageX.unitY.quant_act'
        self.quant_convbn2 = Quant_Conv2d()  # 对应 'stageX.unitY.quant_convbn2'
        if self.downsample is not None:
            self.quant_downsample = Quant_Conv2d()  # 对应 'stageX.unitY.quant_downsample'

        # 将原始的 conv 和 bn 层传递给量化模块
        self.quant_convbn1.set_param(self.conv1, self.bn1)
        self.quant_convbn2.set_param(self.conv2, self.bn3)
        if self.downsample is not None:
            # 假设 downsample 是一个 nn.Sequential 包含 conv 和 bn
            self.quant_downsample.set_param(*downsample)

    def forward_impl(self, x):
        identity = x

        # 第一个卷积和批归一化，并进行量化
        out = self.quant_convbn1(x)  # 对 conv1 和 bn1 进行量化

        # 应用激活量化
        out = self.quant_act(out)  # 对激活进行量化
        out = self.prelu(out)

        # 第二个卷积和批归一化，并进行量化
        out = self.quant_convbn2(out)  # 对 conv2 和 bn3 进行量化

        # 如果需要下采样，则对身份分支进行量化
        if self.downsample is not None:
            identity = self.quant_downsample(x)  # 对 downsample 分支进行量化

        # 残差连接
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

        # 初始卷积和批归一化层
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        # 定义各个阶段，名称与 bit_config.py 中的键一致，例如 'stage1', 'stage2', ...
        self.stage1 = self._make_layer(block, 64, layers[0], stride=2, stage_num=1)
        self.stage2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       stage_num=2)
        self.stage3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       stage_num=3)
        self.stage4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       stage_num=4)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 如果需要，将最后一个批归一化层的权重初始化为0
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, stage_num=1):
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
        for unit_num in range(1, blocks + 1):
            layers.append(
                block(self.inplanes, planes, stride if unit_num == 1 else 1, downsample if unit_num == 1 else None,
                      self.groups, self.base_width, previous_dilation, stage_num, unit_num)
            )
            self.inplanes = planes * block.expansion
            downsample = None

        # 将每个单元命名为 'unit1', 'unit2', ...
        layer = nn.Sequential(*layers)
        for idx, module in enumerate(layer):
            unit_name = f"unit{idx + 1}"
            setattr(layer, unit_name, module)
        return layer

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)  # (B, 512*expansion*7*7)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)  # (B, 512*expansion*7*7) -> (B, 512)
        x = self.features(x)
        return x


def quantize_model_mix(model, bit_config, full_precision_flag=False):
    """
    根据 bit_config 为模型的每一层设置不同的量化位宽。

    参数:
        model (nn.Module): 要量化的模型。
        bit_config (dict): 包含每一层量化配置的字典。
        full_precision_flag (bool): 是否完全使用高精度。
    """

    def get_bit_width(layer_name, bit_config, default_bit=8):
        """
        根据 layer_name 从 bit_config 获取量化位宽。
        """
        if layer_name in bit_config:
            return bit_config[layer_name]
        return default_bit  # 如果找不到，默认使用8位。

    for name, module in model.named_modules():
        # 处理 QuantAct 和 QuantActPreLu 模块
        if isinstance(module, (QuantAct, QuantActPreLu)):
            layer_name = name  # 例如 'stage1.unit1.quant_act'
            act_bit = get_bit_width(layer_name, bit_config, 8)
            module.activation_bit = act_bit
            logging.info(f"Set activation bitwidth for {layer_name}: {act_bit}")

        # 处理 Quant_Conv2d 和 Quant_Linear 模块
        elif isinstance(module, (Quant_Conv2d, Quant_Linear)):
            layer_name = name  # 例如 'stage1.unit1.quant_convbn1'
            weight_bit = get_bit_width(layer_name, bit_config, 8)
            module.weight_bit = weight_bit
            logging.info(f"Set weight bitwidth for {layer_name}: {weight_bit}")

    return model

def quantize_model_mix_i(model, bit_config, full_precision_flag=False):
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
            mods[n] = quantize_model_mix_i(m, bit_config, full_precision_flag)
        return nn.Sequential(mods)

    # Handle other layers (e.g., IBasicBlock, etc.)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:  # Skip normalization layers
                setattr(q_model, attr, quantize_model_mix_i(mod, bit_config, full_precision_flag))
        return q_model



def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    """
    创建指定架构的 IResNet 模型。

    参数:
        arch (str): 模型架构名称。
        block (nn.Module): 块类型。
        layers (list): 每个阶段的块数量。
        pretrained (bool): 是否加载预训练权重。
        progress (bool): 是否显示下载进度。
        **kwargs: 其他参数。

    返回:
        nn.Module: 创建的 IResNet 模型。
    """
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError("Pretrained models are not supported in this implementation.")
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
