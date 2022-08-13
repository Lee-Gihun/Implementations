import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetResBlock, get_conv_layer
from .SparseMax import SparseMax

__all__ = ["UnetrUpBlockSLA", "UnetrBlockSLA"]


class UnetrUpBlockSLA(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        upsample_kernel_size,
        norm_name,
        num_adapters=4,
    ):
        super(UnetrUpBlockSLA, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.sla_transp_conv = SLA(
            self.transp_conv,
            in_channels,
            out_channels,
            upsample_stride,
            num_adapters,
            is_transposed=True,
        )
        self.conv_block = UnetrBlockSLA(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetrBlockSLA(UnetResBlock):
    def __init__(
        self, spatial_dims, in_channels, out_channels, stride, num_adapters=4, **kwargs
    ):
        super(UnetrBlockSLA, self).__init__(
            spatial_dims, in_channels, out_channels, stride=stride, **kwargs
        )
        self.conv1 = SLA(
            self.conv1,
            in_channels,
            out_channels,
            stride=stride,
            num_adapters=num_adapters,
        )
        self.conv2 = SLA(
            self.conv2, out_channels, out_channels, stride=1, num_adapters=num_adapters
        )

        if self.downsample:
            self.conv3 = SLA(
                self.conv3, in_channels, out_channels, stride, num_adapters
            )


class SLA(nn.Module):
    def __init__(
        self,
        module,
        in_channels,
        out_channels,
        stride=1,
        num_adapters=4,
        is_transposed=False,
    ):
        super(SLA, self).__init__()
        self.module = module
        self.adaptive_path = SparseModule(
            in_channels, out_channels, stride, num_adapters, is_transposed
        )

    def forward(self, x):
        out = self.module(x)
        out += self.adaptive_path(x)
        return out


class SparseModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, num_adapters=4, is_transposed=False
    ):
        super(SparseModule, self).__init__()
        self.num_adapters = num_adapters

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_adapters),
            SparseMax(num_adapters),
        )

        if is_transposed:
            self.adapters = nn.ModuleList(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                    )
                    for _ in range(num_adapters)
                ]
            )
        else:
            self.adapters = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                    )
                    for _ in range(num_adapters)
                ]
            )

    def forward(self, x):
        gated_x = self.gate(x)

        out = []
        for adapter_idx, adapter in enumerate(self.adapters):
            gate = gated_x[:, adapter_idx].view(-1, 1, 1, 1)
            out.append(gate * adapter(x))

        out = sum(out)

        return out
