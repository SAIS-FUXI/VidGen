#coding=utf-8
"""
Copy from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List

def is_odd(n):
    return not divisible_by(n, 2)

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        channel_first = True,
        bias = False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        if self.channel_first:
            shape = [1] * x.ndim
            shape[1] = x.size(1)
        else:
            shape = [x.size(-1)]

        x = F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma.view(shape) 
        if self.bias is not None:
            x = x + self.bias.view(shape)
        return x

class CausalConv3d(nn.Module):
    @beartype
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode = 'constant',
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        x = F.pad(x, self.time_causal_padding, mode = pad_mode)
        return self.conv(x)

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = CausalConv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        assert x.ndim == 5
        b, c, t, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) t h w -> qkv b heads c (t h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (t h w) -> b (heads c) t h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        # x: [batch, c, t, h, w]
        assert x.ndim == 5
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)          # [batch, c, t, h, w]
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = q.reshape(b, c, t*h*w)
        q = q.permute(0,2,1)   # b,thw,c
        k = k.reshape(b,c,t*h*w) # b,c,thw
        w_ = torch.bmm(q,k)     # b,thw,thw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,t*h*w)
        w_ = w_.permute(0,2,1)   # b,thw,thw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c, thw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,t,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class AttnBlockST(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm_s = Normalize(in_channels)
        self.q_s = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k_s = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v_s = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out_s = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        self.norm_t = Normalize(in_channels)
        self.q_t = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k_t = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v_t = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out_t = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        # x: [batch, c, t, h, w]
        assert x.ndim == 5
        batch, _, T, H, W = x.shape

        # 1. spatial attn
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        h_ = x
        h_ = self.norm_s(h_)
        q = self.q_s(h_)          # [bt, c, h, w]
        k = self.k_s(h_)
        v = self.v_s(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0,2,1)   # b,thw,c
        k = k.reshape(b,c,h*w) # b,c,thw
        w_ = torch.bmm(q,k)     # b,thw,thw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,thw,thw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c, thw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out_s(h_)

        x = x+h_
        x = rearrange(x, '(b t) c h w -> b c t h w', b=batch)

        # 2. temporal attn
        x = rearrange(x, 'b c t h w -> (b h w) c t')
        h_ = x
        h_ = self.norm_t(h_)
        q = self.q_t(h_)          # [bhw, c, t]
        k = self.k_t(h_)
        v = self.v_t(h_)

        # compute attention
        b, c, l = q.shape
        q = q.permute(0,2,1)        # b,l,c
        w_ = torch.bmm(q,k)         # b,l,l
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0,2,1)   # b,thw,thw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c, thw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = self.proj_out_t(h_)

        x = x+h_
        x = rearrange(x, '(b h w) c t -> b c t h w', b=batch, h=H)
        return x

class CausualAttnBlockST(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        #self.norm_s = Normalize(in_channels)
        self.norm_s = RMSNorm(in_channels, channel_first=True)
        self.q_s = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k_s = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v_s = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out_s = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        #self.norm_t = Normalize(in_channels)
        self.norm_t = RMSNorm(in_channels, channel_first=True)

        self.q_t = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k_t = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v_t = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out_t = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
    
    def _forward_spatial(self, x):
        # x: [batch, c, t, h, w]
        assert x.ndim == 5
        batch, _, T, H, W = x.shape

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        h_ = x
        h_ = self.norm_s(h_)
        q = self.q_s(h_)          # [bt, c, h, w]
        k = self.k_s(h_)
        v = self.v_s(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0,2,1)   # b,thw,c
        k = k.reshape(b,c,h*w) # b,c,thw
        w_ = torch.bmm(q,k)     # b,thw,thw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,thw,thw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c, thw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out_s(h_)

        x = x+h_
        x = rearrange(x, '(b t) c h w -> b c t h w', b=batch)
        return x
    
    def _forward_temporal(self, x, max_temporal_window=None):
        # x: [batch, c, t, h, w]
        assert x.ndim == 5
        batch, _, T, H, W = x.shape

        x = rearrange(x, 'b c t h w -> (b h w) c t')
        h_ = x
        h_ = self.norm_t(h_)
        q = self.q_t(h_)          # [bhw, c, t]
        k = self.k_t(h_)
        v = self.v_t(h_)

        # compute attention
        b, c, l = q.shape
        q = q.permute(0,2,1)        # [bhw, t, c]
        w_ = torch.bmm(q,k)         # [bhw, tq, tk]
        w_ = w_ * (int(c)**(-0.5))

        # causal mask
        mask = torch.full((l, l), torch.finfo(x.dtype).min, device=x.device)
        mask_cond = torch.arange(mask.size(-1), device=x.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        if (max_temporal_window is not None) and (max_temporal_window > 0):
            for i in range(l):
                if (mask[i, :] == 0).sum() > max_temporal_window:
                    delta = (mask[i, :] == 0).sum().item() - max_temporal_window
                    mask[i, :][:delta] = torch.finfo(x.dtype).min

        w_ = w_ + mask
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)   # (bhw tk tq) (first t of k, second of q)
        h_ = torch.bmm(v, w_)     # (bhw c tv)
        h_ = self.proj_out_t(h_)

        x = x + h_
        x = rearrange(x, '(b h w) c t -> b c t h w', b=batch, h=H)
        return x

    def forward(self, x, max_temporal_window=None):
        x = self._forward_spatial(x)
        x = self._forward_temporal(x, max_temporal_window=max_temporal_window)
        return x
    
def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none", "spatial_temporal", "causual_spatial_temporal"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == 'spatial_temporal':
        return AttnBlockST(in_channels)
    elif attn_type == 'causual_spatial_temporal':
        return CausualAttnBlockST(in_channels)
    else:
        return LinAttnBlock(in_channels)

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=(2, 2, 2)):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            self.conv = CausalConv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,)

    def forward(self, x):
        assert x.ndim == 5, 'x.shape should be [b, c, t, h, w]'
        if x.size(2) == 1:
            # the input is image
            x = torch.nn.functional.avg_pool3d(x,
                        kernel_size=(1, self.stride[1], self.stride[2]),
                        stride=(1, self.stride[1], self.stride[2]))
        else:
            # the input is video
            x = torch.nn.functional.avg_pool3d(x, kernel_size=self.stride, stride=self.stride)

        if self.with_conv:
            x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, scale_factor=(2.0, 2.0, 2.0)):
        super().__init__()
        self.with_conv = with_conv
        self.scale_factor = scale_factor
        if self.with_conv:
            self.conv = CausalConv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,)

    def forward(self, x):
        assert x.ndim == 5, 'x.shape should be [b, c, t, h, w]'
        # for video
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        #self.norm1 = Normalize(in_channels)
        self.norm1 = RMSNorm(in_channels, channel_first=True)
        self.conv1 = CausalConv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        #self.norm2 = Normalize(out_channels)
        self.norm2 = RMSNorm(out_channels, channel_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,)
            else:
                self.nin_shortcut = CausalConv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,)

    def forward(self, x, temb):
        h = x                       # [b, c, t, h, w]
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 z_channels, double_z=True,
                 resolution=None,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 spatial_stride=None,
                 temporal_stride=None,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = CausalConv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if len(attn_resolutions) > 0:
                    if curr_res in attn_resolutions:
                        attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                #down.downsample = Downsample(block_in, resamp_with_conv)
                down.downsample = Downsample(block_in, resamp_with_conv,
                                stride=(temporal_stride[i_level], spatial_stride[i_level], spatial_stride[i_level]))
                if len(attn_resolutions) > 0:
                    curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        #self.norm_out = Normalize(block_in)
        self.norm_out = RMSNorm(block_in, channel_first=True)
        self.conv_out = CausalConv3d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,)

    def forward(self, x, max_temporal_window=None):
        """
        inputs:
            x: [batch, 3, T, H, W]
        """
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h, max_temporal_window=max_temporal_window)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
    def forward_debug(self, x, max_temporal_window=None):
        """
        inputs:
            x: [batch, 3, T, H, W]
        """
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        #h = hs[-1]
        # h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h, max_temporal_window=max_temporal_window)
        # h = self.mid.block_2(h, temb)

        # # end
        # h = self.norm_out(h)
        # h = nonlinearity(h)
        # h = self.conv_out(h)
        return hs

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution=None,
                 spatial_stride=None,
                 temporal_stride=None,
                 z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", 
                 **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        if len(attn_resolutions) > 0:
            curr_res = resolution // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = CausalConv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if len(attn_resolutions) > 0:
                    if curr_res in attn_resolutions:
                        attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv,
                                scale_factor=(float(temporal_stride[i_level-1]), float(spatial_stride[i_level-1]), float(spatial_stride[i_level-1])))
                if len(attn_resolutions) > 0:
                    curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        if not ignorekwargs.get('revise_norm_bug', False):
            # bug, groupnorm 对causal有影响，需要用rmsnorm替代
            self.norm_out = Normalize(block_in)
        else:
            self.norm_out = RMSNorm(block_in, channel_first=True)
        self.conv_out = CausalConv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,)

    def forward(self, z, max_temporal_window=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h, max_temporal_window=max_temporal_window)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
    def forward_debug(self, z, max_temporal_window=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        hs = [h]

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h, max_temporal_window=max_temporal_window)
        h = self.mid.block_2(h, temb)
        hs.append(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                hs.append(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        hs.append(h)
        return hs
