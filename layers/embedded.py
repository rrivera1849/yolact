
import torch.nn as nn

from math import gcd

class MobileNetV1ConvBlock(nn.Module):
    """Builds a MobileNetV1 convolutional block. 
       This block is used as a part of the YOLACT Embedded mode to lower the number 
       of parameters and thus speed up the computation.

       Note that we're not applying BatchNorm or ReLU6 after the pointwise convolution. 
       We leave it to each module to decide whether they want to apply it or not. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(MobileNetV1ConvBlock, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, 
                                        kernel_size, stride, padding, 
                                        groups=gcd(in_channels, out_channels), 
                                        bias=False, **kwargs)

        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6()

        # Is a Bias needed here? 
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.activation(self.bn_1(out))
        out = self.pointwise_conv(out)

        return out


