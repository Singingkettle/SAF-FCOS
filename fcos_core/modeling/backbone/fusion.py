import torch
from torch import nn

from fcos_core.layers import Conv2d


class ATTBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, padding):
        super(ATTBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class ATTMix(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_sizes, paddings):
        super(ATTMix, self).__init__()
        self.stages = []
        self.div_constant = len(kernel_sizes)
        for i in range(len(kernel_sizes)):
            name = "ATT%dX%d" % (i, i)
            module = ATTBlock(input_channel, output_channel, kernel_sizes[i], paddings[i])
            self.add_module(name, module)
            self.stages.append(name)

    def forward(self, x):
        outputs = []
        for stage_name in self.stages:
            x_ = getattr(self, stage_name)(x)
            outputs.append(x_)

        x = torch.cat(outputs, 1)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.div(x, self.div_constant)
        return x


class FusionAdd(nn.Module):
    def __init__(self, input_channels, cfg):
        super(FusionAdd, self).__init__()

    def forward(self, im_x, ra_x):
        x = torch.add(im_x, ra_x)

        return x


class FusionConcat(nn.Module):
    def __init__(self, input_channels, cfg):
        super(FusionConcat, self).__init__()
        self.fusion_down_sample = Conv2d(in_channels=input_channels * 2, out_channels=input_channels,
                                         kernel_size=1, padding=0)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, im_x, ra_x):
        x = torch.cat((im_x, ra_x), 1)
        x = self.fusion_down_sample(x)
        return x


class FusionMul(nn.Module):
    def __init__(self, input_channels, cfg):
        super(FusionMul, self).__init__()

    def forward(self, im_x, ra_x):
        x = torch.mul(im_x, ra_x)
        return x


class FusionAtt1X1(nn.Module):
    def __init__(self, input_channels, cfg):
        super(FusionAtt1X1, self).__init__()
        self.att = ATTBlock(input_channels, 1, 1, 0)

    def forward(self, im_x, ra_x):
        ra_x = self.att(ra_x)
        x = torch.mul(im_x, ra_x)
        return x


class FusionAtt3X3(nn.Module):
    def __init__(self, input_channels, cfg):
        super(FusionAtt3X3, self).__init__()
        self.att = ATTBlock(input_channels, 1, 3, 1)

    def forward(self, im_x, ra_x):
        ra_x = self.att(ra_x)
        x = torch.mul(im_x, ra_x)
        return x


class FusionAtt5X5(nn.Module):
    def __init__(self, input_channels, cfg):
        super(FusionAtt5X5, self).__init__()
        self.att = ATTBlock(input_channels, 1, 5, 2)

    def forward(self, im_x, ra_x):
        ra_x = self.att(ra_x)
        x = torch.mul(im_x, ra_x)
        return x


class FusionAtt7X7(nn.Module):
    def __init__(self, input_channels, cfg):
        super(FusionAtt7X7, self).__init__()
        self.att = ATTBlock(input_channels, 1, 7, 3)

    def forward(self, im_x, ra_x):
        ra_x = self.att(ra_x)
        x = torch.mul(im_x, ra_x)
        return x


class FusionAttMix(nn.Module):
    def __init__(self, input_channels, cfg):
        super(FusionAttMix, self).__init__()
        self.att = ATTMix(input_channels, 1, cfg.MODEL.BACKBONE.FUSION_MIX_KERNEL_SIZES,
                          cfg.MODEL.BACKBONE.FUSION_MIX_PADDINGS)

    def forward(self, im_x, ra_x):
        ra_x = self.att(ra_x)
        x = torch.mul(im_x, ra_x)
        return x
