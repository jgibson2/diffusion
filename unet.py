"""
Adapted from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/vision/unet.py
Uses some tricks from https://arxiv.org/pdf/2105.05233.pdf, including:
 * multiple attention heads
 * attention at multiple levels
 * rescaling residual connections by 1/sqrt(2)
"""
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        attention_layers: Indices of layers (0-based) for which to use self-attention. The corresponding up layer will also use attention.
        attention_heads: number of heads to use for attention
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
            self,
            input_channels: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            attention_layers: Tuple[int, ...] = (1, 2, 3),
            attention_heads: int = 4,
            time_enc_dim: int = -1,
            bilinear: bool = False,
            predict_scalars: bool = False
    ):

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for i in range(num_layers - 1):
            layers.append(Down(feats,
                               feats * 2,
                               attention_heads=attention_heads if i in attention_layers else -1,
                               time_enc_dim=time_enc_dim,
                               ))
            feats *= 2

        for i in range(num_layers - 2, -1, -1):
            layers.append(Up(feats,
                             feats // 2,
                             attention_heads=attention_heads if i in attention_layers else -1,
                             time_enc_dim=time_enc_dim,
                             bilinear=bilinear,
                             ))
            feats //= 2

        layers.append(nn.Conv2d(feats, input_channels, kernel_size=(1, 1)))
        self.layers = nn.ModuleList(layers)

        self.predict_scalars = predict_scalars
        if self.predict_scalars:
            self.scalar_net = nn.Sequential(
                nn.Conv2d(feats, 1, kernel_size=(3, 3)),
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten()
            )

    def forward(self, x, emb=None):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1], emb=emb))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], (1.0 / np.sqrt(2.0)) * xi[-2 - i], emb=emb)
        output = self.layers[-1](xi[-1])
        if self.predict_scalars:
            scalars = self.scalar_net(xi[-1])
            return output, scalars
        else:
            return output


class DoubleConv(nn.Module):
    """[ Conv2d => Groupnorm => ReLU ] x 2 with optional attention."""

    def __init__(self, in_ch: int, out_ch: int, attention_heads=-1, time_enc_dim=-1, dropout=0.0):
        super().__init__()
        in_modules = (
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )
        out_modules = (
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )
        self.in_net = nn.Sequential(*in_modules)
        self.out_net = nn.Sequential(*out_modules)
        self.use_attention = attention_heads != -1
        self.use_time_enc = time_enc_dim != -1
        if self.use_attention:
            self.attn = AttentionBlock(out_ch, num_heads=attention_heads)
        if self.use_time_enc:
            self.time_enc_net = nn.Linear(time_enc_dim, out_ch)
        self.use_dropout = dropout != 0
        if self.use_dropout:
            self.use_dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        x = self.in_net(x)
        if emb is not None:
            if not self.use_time_enc:
                raise ValueError("embedding passed but no encoding net present")
            emb_out = self.time_enc_net(emb.type(x.dtype))
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out
        if self.use_attention:
            x = self.attn(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.out_net(x)
        return x


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int, attention_heads=-1, time_enc_dim=-1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, attention_heads=attention_heads, time_enc_dim=time_enc_dim)

    def forward(self, x, emb=None):
        x = self.pool(x)
        x = self.conv(x, emb=emb)
        return x


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, attention_heads=-1, time_enc_dim=-1, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=(1, 1)),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=(2, 2), stride=(2, 2))

        self.conv = DoubleConv(in_ch, out_ch, attention_heads=attention_heads, time_enc_dim=time_enc_dim)

    def forward(self, x1, x2, emb=None):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, emb=emb)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=(1,))
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=(1,))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))

        bs, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.num_heads, ch, length),
            (k * scale).view(bs * self.num_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.num_heads, ch, length))

        a = a.reshape(bs, -1, length)

        a = self.proj_out(a)
        return (x + a).reshape(b, c, *spatial)
