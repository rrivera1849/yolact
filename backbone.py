
import collections
import math
import pickle
import re
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dcn_v2 import DCN
except ImportError:
    def DCN(*args, **kwdargs):
        raise Exception('DCN could not be imported. If you want to use YOLACT++ models, compile DCN. Check the README for instructions.')

def add_to_outputs(backbone, outputs, out, idx, ignore=False):
    """Adds `out` to the `outputs` list if it is in model.selected_layers.
       If there is no such attribute, will append every value.
    """
    if ignore:
        outputs.append(out)
        return

    if hasattr(backbone, 'selected_layers'):
        if idx in backbone.selected_layers:
            outputs.append(out)
    else:
        outputs.append(out)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ Adapted from torchvision.models.resnet """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """ Adapted from torchvision.models.resnet """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ Adapted from torchvision.models.resnet """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_dcn=False):
        super(BasicBlock, self).__init__()

        if use_dcn:
            raise ValueError("DCN is not supported for ResNet's smaller than ResNet-50")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1, use_dcn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        if use_dcn:
            self.conv2 = DCN(planes, planes, kernel_size=3, stride=stride,
                                padding=dilation, dilation=dilation, deformable_groups=1)
            self.conv2.bias.data.zero_()
            self.conv2.conv_offset_mask.weight.data.zero_()
            self.conv2.conv_offset_mask.bias.data.zero_()
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=dilation, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, dcn_layers=[0, 0, 0, 0], dcn_interval=1, atrous_layers=[], block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self._make_layer(block, 64, layers[0], dcn_layers=dcn_layers[0], dcn_interval=dcn_interval)
        self._make_layer(block, 128, layers[1], stride=2, dcn_layers=dcn_layers[1], dcn_interval=dcn_interval)
        self._make_layer(block, 256, layers[2], stride=2, dcn_layers=dcn_layers[2], dcn_interval=dcn_interval)
        self._make_layer(block, 512, layers[3], stride=2, dcn_layers=dcn_layers[3], dcn_interval=dcn_interval)

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
    
    def _make_layer(self, block, planes, blocks, stride=1, dcn_layers=0, dcn_interval=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        # This is actually just to create the connection between layers, and not necessarily to
        # downsample. Even if the second condition is met, it only downsamples when stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          dilation=self.dilation),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        use_dcn = (dcn_layers >= blocks)
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self.norm_layer, dilation=self.dilation, use_dcn=use_dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_dcn = ((i+dcn_layers) >= blocks) and (i % dcn_interval == 0)
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer, use_dcn=use_dcn))
        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def forward(self, x):
        """ Returns a list of convouts for each layer. """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            add_to_outputs(self, outs, x, i, ignore=False)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx-1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)




class ResNetBackboneGN(ResNetBackbone):

    def __init__(self, layers, num_groups=32):
        super().__init__(layers, norm_layer=lambda x: nn.GroupNorm(num_groups, x))

    def init_backbone(self, path):
        """ The path here comes from detectron. So we load it differently. """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1') # From the detectron source
            state_dict = state_dict['blobs']
        
        our_state_dict_keys = list(self.state_dict().keys())
        new_state_dict = {}
    
        gn_trans     = lambda x: ('gn_s' if x == 'weight' else 'gn_b')
        layeridx2res = lambda x: 'res' + str(int(x)+2)
        block2branch = lambda x: 'branch2' + ('a', 'b', 'c')[int(x[-1:])-1]

        # Transcribe each Detectron weights name to a Yolact weights name
        for key in our_state_dict_keys:
            parts = key.split('.')
            transcribed_key = ''

            if (parts[0] == 'conv1'):
                transcribed_key = 'conv1_w'
            elif (parts[0] == 'bn1'):
                transcribed_key = 'conv1_' + gn_trans(parts[1])
            elif (parts[0] == 'layers'):
                if int(parts[1]) >= self.num_base_layers: continue

                transcribed_key = layeridx2res(parts[1])
                transcribed_key += '_' + parts[2] + '_'

                if parts[3] == 'downsample':
                    transcribed_key += 'branch1_'
                    
                    if parts[4] == '0':
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[5])
                else:
                    transcribed_key += block2branch(parts[3]) + '_'

                    if 'conv' in parts[3]:
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[4])

            new_state_dict[key] = torch.Tensor(state_dict[transcribed_key])
        
        # strict=False because we may have extra unitialized layers at this point
        self.load_state_dict(new_state_dict, strict=False)







def darknetconvlayer(in_channels, out_channels, *args, **kwdargs):
    """
    Implements a conv, activation, then batch norm.
    Arguments are passed into the conv layer.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwdargs, bias=False),
        nn.BatchNorm2d(out_channels),
        # Darknet uses 0.1 here.
        # See https://github.com/pjreddie/darknet/blob/680d3bde1924c8ee2d1c1dea54d3e56a05ca9a26/src/activations.h#L39
        nn.LeakyReLU(0.1, inplace=True)
    )

class DarkNetBlock(nn.Module):
    """ Note: channels is the lesser of the two. The output will be expansion * channels. """

    expansion = 2

    def __init__(self, in_channels, channels):
        super().__init__()

        self.conv1 = darknetconvlayer(in_channels, channels,                  kernel_size=1)
        self.conv2 = darknetconvlayer(channels,    channels * self.expansion, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x




class DarkNetBackbone(nn.Module):
    """
    An implementation of YOLOv3's Darnet53 in
    https://pjreddie.com/media/files/papers/YOLOv3.pdf

    This is based off of the implementation of Resnet above.
    """

    def __init__(self, layers=[1, 2, 8, 8, 4], block=DarkNetBlock):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        
        self._preconv = darknetconvlayer(3, 32, kernel_size=3, padding=1)
        self.in_channels = 32
        
        self._make_layer(block, 32,  layers[0])
        self._make_layer(block, 64,  layers[1])
        self._make_layer(block, 128, layers[2])
        self._make_layer(block, 256, layers[3])
        self._make_layer(block, 512, layers[4])

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
    
    def _make_layer(self, block, channels, num_blocks, stride=2):
        """ Here one layer means a string of n blocks. """
        layer_list = []

        # The downsample layer
        layer_list.append(
            darknetconvlayer(self.in_channels, channels * block.expansion,
                             kernel_size=3, padding=1, stride=stride))

        # Each block inputs channels and outputs channels * expansion
        self.in_channels = channels * block.expansion
        layer_list += [block(self.in_channels, channels) for _ in range(num_blocks)]

        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layer_list))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """

        x = self._preconv(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            add_to_outputs(self, outs, x, i, ignore=False)

        return tuple(outs)

    def add_layer(self, conv_channels=1024, stride=2, depth=1, block=DarkNetBlock):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, num_blocks=depth, stride=stride)
    
    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(torch.load(path), strict=False)





class VGGBackbone(nn.Module):
    """
    Args:
        - cfg: A list of layers given as lists. Layers can be either 'M' signifying
                a max pooling layer, a number signifying that many feature maps in
                a conv layer, or a tuple of 'M' or a number and a kwdargs dict to pass
                into the function that creates the layer (e.g. nn.MaxPool2d for 'M').
        - extra_args: A list of lists of arguments to pass into add_layer.
        - norm_layers: Layers indices that need to pass through an l2norm layer.
    """

    def __init__(self, cfg, extra_args=[], norm_layers=[]):
        super().__init__()
        
        self.channels = []
        self.layers = nn.ModuleList()
        self.in_channels = 3
        self.extra_args = list(reversed(extra_args)) # So I can use it as a stack

        # Keeps track of what the corresponding key will be in the state dict of the
        # pretrained model. For instance, layers.0.2 for us is 2 for the pretrained
        # model but layers.1.1 is 5.
        self.total_layer_count = 0
        self.state_dict_lookup = {}

        for idx, layer_cfg in enumerate(cfg):
            self._make_layer(layer_cfg)

        self.norms = nn.ModuleList([nn.BatchNorm2d(self.channels[l]) for l in norm_layers])
        self.norm_lookup = {l: idx for idx, l in enumerate(norm_layers)}

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, cfg):
        """
        Each layer is a sequence of conv layers usually preceded by a max pooling.
        Adapted from torchvision.models.vgg.make_layers.
        """

        layers = []

        for v in cfg:
            # VGG in SSD requires some special layers, so allow layers to be tuples of
            # (<M or num_features>, kwdargs dict)
            args = None
            if isinstance(v, tuple):
                args = v[1]
                v = v[0]

            # v should be either M or a number
            if v == 'M':
                # Set default arguments
                if args is None:
                    args = {'kernel_size': 2, 'stride': 2}

                layers.append(nn.MaxPool2d(**args))
            else:
                # See the comment in __init__ for an explanation of this
                cur_layer_idx = self.total_layer_count + len(layers)
                self.state_dict_lookup[cur_layer_idx] = '%d.%d' % (len(self.layers), len(layers))

                # Set default arguments
                if args is None:
                    args = {'kernel_size': 3, 'padding': 1}

                # Add the layers
                layers.append(nn.Conv2d(self.in_channels, v, **args))
                layers.append(nn.ReLU(inplace=True))
                self.in_channels = v
        
        self.total_layer_count += len(layers)
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply an l2norm module to the selected layers
            # Note that this differs from the original implemenetation
            if idx in self.norm_lookup:
                x = self.norms[self.norm_lookup[idx]](x)

            add_to_outputs(self, outs, x, idx, ignore=False)
        
        return tuple(outs)

    def transform_key(self, k):
        """ Transform e.g. features.24.bias to layers.4.1.bias """
        vals = k.split('.')
        layerIdx = self.state_dict_lookup[int(vals[0])]
        return 'layers.%s.%s' % (layerIdx, vals[1])

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        state_dict = OrderedDict([(self.transform_key(k), v) for k,v in state_dict.items()])

        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=128, downsample=2):
        """ Add a downsample layer to the backbone as per what SSD does. """
        if len(self.extra_args) > 0:
            conv_channels, downsample = self.extra_args.pop()
        
        padding = 1 if downsample > 1 else 0
        
        layer = nn.Sequential(
            nn.Conv2d(self.in_channels, conv_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels*2, kernel_size=3, stride=downsample, padding=padding),
            nn.ReLU(inplace=True)
        )

        self.in_channels = conv_channels*2
        self.channels.append(self.in_channels)
        self.layers.append(layer)
        

##################################################
# YOLACT Embedded Backbones
##################################################

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


class MobileNetV2Backbone(nn.Module):
    """
    Adapted from torchvision.models.mobilenet.MobileNetV2
    """
    def __init__(self,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=InvertedResidual):
        super(MobileNetV2Backbone, self).__init__()

        input_channel = 32
        last_channel = 1280
        self.channels = []

        self.layers = nn.ModuleList()

        if inverted_residual_setting is None:
            raise ValueError("Must provide inverted_residual_setting where each element is a list "
                             "that represents the MobileNetV2 t,c,n,s values for that layer.")

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.layers.append(ConvBNReLU(3, input_channel, stride=2))
        self.channels.append(input_channel)

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            input_channel = self._make_layer(input_channel, width_mult, round_nearest, t, c, n, s, block)
            self.channels.append(input_channel)

        # building last several layers
        self.layers.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.channels.append(self.last_channel)

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]


    def _make_layer(self, input_channel, width_mult, round_nearest, t, c, n, s, block):
        """A layer is a combination of inverted residual blocks"""
        layers = []
        output_channel = _make_divisible(c * width_mult, round_nearest)

        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel

        self.layers.append(nn.Sequential(*layers))
        return input_channel


    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        outs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
            add_to_outputs(self, outs, x, idx, ignore=False)
        
        return tuple(outs)


    def add_layer(self, conv_channels=1280, t=1, c=1280, n=1, s=2):
        """TODO: Need to make sure that this works as intended.
        """
        self._make_layer(conv_channels, 1.0, 8, t, c, n, s, InvertedResidual)


    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        checkpoint = torch.load(path)

        checkpoint.pop('classifier.1.weight')
        checkpoint.pop('classifier.1.bias')

        checkpoint_keys = list(checkpoint.keys())
        assert len(checkpoint_keys) == len(self.state_dict())
        transform_dict = dict(zip(checkpoint, list(self.state_dict().keys())))

        state_dict = OrderedDict([(transform_dict[k], v) for k,v in checkpoint.items()])

        self.load_state_dict(state_dict, strict=True)


class MobileNetV3Backbone(nn.Module):
    def __init__(self, cfg, width_mult=1.0, block=InvertedResidualV3):
        super(MobileNetV3Backbone, self).__init__()
        assert len(cfg) >= 0 and len(cfg[0]) == 6

        self.layers = nn.ModuleList()
        self.channels = []

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        self.layers.append(Conv3x3BNSwish(3, input_channel, stride=2))
        self.channels.append(input_channel)

        # building inverted residual blocks
        for k, t, c, se, hs, s in cfg:
            input_channel, expansion_size = self._make_layer(input_channel, width_mult, 8, k, t, c, se, hs, s, block)
            self.channels.append(input_channel)

        self.layers.append(Conv1x1BNSwish(input_channel, expansion_size))
        self.channels.append(expansion_size)

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]


    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        outs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
            add_to_outputs(self, outs, x, idx, ignore=False)
        
        return tuple(outs)


    def _make_layer(self, input_channel, width_mult, round_nearest, k, t, c, se, hs, s, block):
        """A layer is a combination of inverted residual blocks"""
        output_channel = _make_divisible(c * width_mult, round_nearest)
        expansion_size = _make_divisible(input_channel * t, round_nearest)

        self.layers.append(block(input_channel, expansion_size, output_channel, s, k, se, hs))
        return output_channel, expansion_size


    def add_layer(self, conv_channels=960, k=5, t=6, c=1280, se=1, hs=1, s=2):
        self._make_layer(conv_channels, 1.0, 8, k, t, c, se, hs, s, InvertedResidualV3)


    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        checkpoint = torch.load(path)

        checkpoint.pop('classifier.0.weight')
        checkpoint.pop('classifier.0.bias')
        checkpoint.pop('classifier.3.weight')
        checkpoint.pop('classifier.3.bias')

        checkpoint_keys = list(checkpoint.keys())
        assert len(checkpoint_keys) == len(self.state_dict())
        transform_dict = dict(zip(checkpoint, list(self.state_dict().keys())))

        state_dict = OrderedDict([(transform_dict[k], v) for k,v in checkpoint.items()])

        self.load_state_dict(state_dict, strict=True)


class HarDNetBackbone(nn.Module):
    def __init__(self, model_name):
        super(HarDNetBackbone, self).__init__()

        github = 'PingoLH/Pytorch-HarDNet'

        model_list = torch.hub.list(github, model_name)
        assert model_name in model_list

        self.model = torch.hub.load(github, model_name, pretrained=True)
        self.model.base = self.model.base[:len(self.model.base) - 1]

        self.channels = [32, 64, 64, 124, 128, 128, 262, 256, 328, 320, 320, 654, 640, 640, 782, 1024]

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]


    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []

        for idx, layer in enumerate(self.model.base):
            x = layer(x)
            
            add_to_outputs(self, outs, x, idx, ignore=False)
        
        return tuple(outs)


    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        # Weights are being loaded from torch.hub
        pass

    def add_layer(self):
        pass


def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """
    backbone = cfg.type(*cfg.args)

    # Add downsampling layers until we reach the number we need
    num_layers = max(cfg.selected_layers) + 1

    if isinstance(backbone, HarDNetBackbone):
        while len(backbone.model.base) < num_layers:
            backbone.add_layer()
    else:
        while len(backbone.layers) < num_layers:
            backbone.add_layer()

    return backbone

# # Just some quick testing code
# if __name__ == "__main__":
    # mobilenetv3_large_arch = [
        # # k, t, c, SE, HS, s 
        # [3,   1,  16, 0, 0, 1],
        # [3,   4,  24, 0, 0, 2],
        # [3,   3,  24, 0, 0, 1],
        # [5,   3,  40, 1, 0, 2],
        # [5,   3,  40, 1, 0, 1],
        # [5,   3,  40, 1, 0, 1],
        # [3,   6,  80, 0, 1, 2],
        # [3, 2.5,  80, 0, 1, 1],
        # [3, 2.3,  80, 0, 1, 1],
        # [3, 2.3,  80, 0, 1, 1],
        # [3,   6, 112, 1, 1, 1],
        # [3,   6, 112, 1, 1, 1],
        # [5,   6, 160, 1, 1, 2],
        # [5,   6, 160, 1, 1, 1],
        # [5,   6, 160, 1, 1, 1]
    # ]

    # print("Loading MobileNetV3")
    # model = MobileNetV3Backbone(mobilenetv3_large_arch)
    # model.init_backbone("./weights/mobilenetv3-large-1cd25616.pth")

    # mobilenetv2_arch = [
        # # t, c, n, s
        # [1, 16, 1, 1],
        # [6, 24, 2, 2],
        # [6, 32, 3, 2],
        # [6, 64, 4, 2],
        # [6, 96, 3, 1],
        # [6, 160, 3, 2],
        # [6, 320, 1, 1],
    # ]

    # print("Loading MobileNetV2")
    # model = MobileNetV2Backbone(1.0, mobilenetv2_arch, 8)
    # model.init_backbone("./weights/mobilenet_v2-b0353104.pth")
