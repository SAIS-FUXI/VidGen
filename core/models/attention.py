#coding=utf-8
import torch
import torch.nn as nn
try:
    import xformers.ops
except:
    print('install xformers first')
from math import pi
from einops import rearrange, repeat
import torch.nn.functional as F

from core.models.pos_embed import rotate_half


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

class RMSNorm(nn.Module):
    def __init__(self, dim, channel_first = False, bias = False):
        super().__init__()
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        dtype = x.dtype
        if self.channel_first:
            shape = [1] * x.ndim
            shape[1] = x.size(1)
        else:
            shape = [x.size(-1)]
        x = F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma.view(shape) 
        if self.bias is not None:
            x = x + self.bias.view(shape)
        x = x.to(dtype)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0.,
                 qk_norm: bool = False, 
                 norm_layer: nn.Module = RMSNorm,):
        super(CrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        assert mask is not None

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        # add qk-norm
        q, k = self.q_norm(q), self.k_norm(k)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                     # [B, #head, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.to(torch.float32)       # # translate attn to float32
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LlamaMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class AttentionRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        #norm_layer: nn.Module = nn.LayerNorm,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        B, N, C = x.shape
        assert freqs_cos.ndim == 3 and freqs_cos.size(0) == B and freqs_cos.size(1) == N

        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)                                     # [B, #heads, N, D]
        q, k = self.q_norm(q), self.k_norm(k)

        freqs_cos = freqs_cos.unsqueeze(1).to(x)
        freqs_sin = freqs_sin.unsqueeze(1).to(x)

        # q, k: [B, num_heads, N, head_dim]
        q = q * freqs_cos + rotate_half(q) * freqs_sin
        k = k * freqs_cos + rotate_half(k) * freqs_sin

        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B, #head, N, N]
        attn = attn.to(torch.float32)   # translate attn to float32
        if mask is not None:
            assert (mask.shape == (B, N, N)) or (mask.shape == (1, N, N))
            attn = attn + mask.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)           # cast back attn to original dtype
        attn = self.attn_drop(attn)
        x = attn @ v

        x_output_shape = (B, N, C)
        x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# def rotate_half(x):
#     x = rearrange(x, '... (d r) -> ... d r', r=2)
#     x1, x2 = x.unbind(dim=-1)
#     x = torch.stack((-x2, x1), dim=-1)
#     return rearrange(x, '... d r -> ... (d r)')

def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)
