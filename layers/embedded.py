
import collections
import math
import re
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo


class h_sigmoid(nn.Module):
    """
    Adapted from https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """
    Adapted from https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.h_sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.h_sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class ConvBNAct(nn.Sequential):
    """
    Adapted from torchvision.models.mobilenet.ConvBNReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, activation=nn.ReLU6(inplace=True)):
        padding = (kernel_size - 1) // 2
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            activation
        )


ConvBN = partial(ConvBNAct, activation=nn.Identity())
ConvBNReLU = partial(ConvBNAct)
Conv3x3BNSwish = partial(ConvBNAct, kernel_size=3, activation=h_swish(inplace=True))
Conv1x1BNSwish = partial(ConvBNAct, kernel_size=1, stride=1, activation=h_swish(inplace=True))


class MobileNetV1ConvBlock(nn.Module):
    """Builds a MobileNetV1 convolutional block. 
       This block is used as a part of the YOLACT Embedded mode to lower the number 
       of parameters and thus speed up the computation.

       Note that we're not applying BatchNorm or ReLU after the pointwise convolution. 
       We leave it to each module to decide whether they want to apply it or not. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(MobileNetV1ConvBlock, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, 
                                        kernel_size, stride, padding, 
                                        groups=math.gcd(in_channels, out_channels), 
                                        bias=False, **kwargs)

        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.activation(self.bn_1(out))
        out = self.pointwise_conv(out)

        return out


def _make_divisible(v, divisor, min_value=None):
    """
    Adapted from torchvision.models.mobilenet._make_divisible.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """
    Adapted from https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
    """
    def __init__(self, channel, reduction=4):
        super(SqueezeExcite, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        intermediate = _make_divisible(channel // reduction, divisor=8)
        self.fc = nn.Sequential(
                nn.Linear(channel, intermediate),
                nn.ReLU(inplace=True),
                nn.Linear(intermediate, channel))


    def forward(self, x):
        B, C, _, _ = x.size()
        weights = self.avg_pool(x).view(B, C)
        weights = self.fc(weights).view(B, C, 1, 1)

        return x * weights


class InvertedResidual(nn.Module):
    """
    Adapted from torchvision.models.mobilenet.InvertedResidual
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualV3(nn.Module):
    """
    Adapted from https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py

    Inverted Residual as described in https://arxiv.org/pdf/1905.02244.pdf
    """
    def __init__(self, inp, hidden_dim, oup, stride, kernel_size, SE=False, HS=False):
        super(InvertedResidualV3, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                ConvBN(hidden_dim, hidden_dim, kernel_size, stride, groups=hidden_dim),
                h_swish() if HS else nn.ReLU(inplace=True),
                SqueezeExcite(hidden_dim) if SE else nn.Identity(),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                ConvBN(inp, hidden_dim, 1, 1),
                h_swish() if HS else nn.ReLU(inplace=True),

                # dw
                ConvBN(hidden_dim, hidden_dim, kernel_size, stride, groups=hidden_dim),
                SqueezeExcite(hidden_dim) if SE else nn.Identity(),
                h_swish() if HS else nn.ReLU(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

##################################################
# Usefule Parameters / Layers for EfficientNet Implementation
# Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
# Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
##################################################

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'width_divisor', 'drop_connect_rate', 'image_size'])


BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


class BlockDecoder(object):
    """Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """
    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value


        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])


    @staticmethod
    def _encode_block_string(block):

        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)


    @staticmethod
    def decode(string_list):

        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):

        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def round_filters(filters, global_params):
    """Calculates the appropriate filters given the width coefficient and 
       specified depth divisor.
       
       Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """
    new_filters = _make_divisible(filters * global_params.width_coefficient,
                                  global_params.width_divisor)

    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculates the appropriate number of repeats given the depth coefficient.

       Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """
    multiplier = global_params.depth_coefficient

    return int(math.ceil(multiplier * repeats))


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.width_divisor
    min_depth = None
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ This is the same function as dropout.
    """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x):
    """Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """
    if isinstance(x, int): return x, x
    if isinstance(x, list) or isinstance(x, tuple): return x
    else: raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """
    if input_image_size is None: return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def calculate_padding_size(ih, iw, kh, kw, sh, sw, dh, dw):
    """Computes the padding size given the image dimensions, kernel dimensions, and 
       stride along each dimension.
    """
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
    pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
    pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)

    return pad_h, pad_w


class Conv2dDynamicSamePadding(nn.Conv2d):
    """Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        dh, dw = self.dilation[0], self.dilation[1]

        pad_h, pad_w = calculate_padding_size(ih, iw, kh, kw, sh, sw, dh, dw)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert image_size is not None

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        dh, dw = self.dilation[0], self.dilation[1]

        pad_h, pad_w = calculate_padding_size(ih, iw, kh, kw, sh, sw, dh, dw)

        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class MBConvBlock(nn.Module):
    """Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
    """
    def __init__(self, block_args, global_params, image_size=None):
        super(MBConvBlock, self).__init__()

        self._block_args = block_args
        self._bn_momentum = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        # Expansion Phase
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio

        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(inp, oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(oup, momentum=self._bn_momentum, eps=self._bn_eps)
            image_size = calculate_output_image_size(image_size, 1)

        # Detphwise Convolution Phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(oup, oup, kernel_size=k, stride=s, groups=oup, bias=False)
        self._bn1 = nn.BatchNorm2d(oup, momentum=self._bn_momentum, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze Excitation Phase, optional
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(oup, num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(num_squeezed_channels, oup, kernel_size=1)

        # Output Phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(oup, final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(final_oup, momentum=self._bn_momentum, eps=self._bn_eps)
        self._swish = swish()

    
    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))

        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters

        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs

        return x
