

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(x, size, upsample_mode):
    """Adapted from: 
       https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/merge_cells.py
    """
    if x.shape[-2:] == size:
        return x
    elif x.shape[-2:] < size:
        return F.interpolate(x, size=size, mode=upsample_mode, align_corners=False)
    else:
        # If the shapes aren't evenly divisible we do enough padding so that they 
        # are and then apply the maxpool.
        if (x.shape[-2] % size[-2] != 0 and x.shape[-1] % size[-1] != 0):
            multiple = (x.shape[-2] // size[0]) + 1
            pad_amount = (size[0] * multiple) - x.shape[-2]
            pad = (0, pad_amount, 0, pad_amount)

            x = F.pad(x, pad, mode="constant", value=0)

        kernel_size = x.shape[-1] // size[-1]
        x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
        return x


class SumCell(nn.Module):
    """Implements a NAS-FPN SumCell as defined in:
       https://arxiv.org/abs/1904.07392
    """
    def __init__(self, in_channels, out_channels, upsample_mode):
        super().__init__()
        self.upsample_mode = upsample_mode

        self.out_conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x1, x2, resize_size):
        
        resize_size = torch.Size((size.item() for size in resize_size))
        x1 = resize(x1, resize_size, self.upsample_mode)
        x2 = resize(x2, resize_size, self.upsample_mode)

        out = x1 + x2
        out = self.out_conv(out)

        return out


class GlobalPoolingCell(nn.Module):
    """Implements a NAS-FPN GlobalPoolingCell as defined in:
       https://arxiv.org/abs/1904.07392
    """
    def __init__(self, in_channels, out_channels, upsample_mode):
        super().__init__()
        self.upsample_mode = upsample_mode

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x1, x2, resize_size):

        resize_size = torch.Size((size.item() for size in resize_size))
        x1 = resize(x1, resize_size, self.upsample_mode)
        x2 = resize(x2, resize_size, self.upsample_mode)

        x2_att = torch.sigmoid(self.global_pool(x2))
        out = x2 + x2_att * x1

        return out
