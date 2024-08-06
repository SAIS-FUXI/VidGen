# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from einops import rearrange
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs//s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)

class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption

class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        assert x.ndim == 5
        # padding
        B, _, T, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        Wt, Wh, Ww = x.size(2), x.size(3), x.size(4)

        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(B, self.embed_dim, Wt, Wh, Ww)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x, Wt, Wh, Ww
    

from transformers import AutoProcessor, CLIPVisionModel
class FrozenCLIPEmbedder(nn.Module):
    def __init__(self, path="openai/clip-vit-huge-patch14", device="cuda"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = CLIPVisionModel.from_pretrained(path, device_map=device)
        self.device = device
        self._freeze()

    def _freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        outputs = self.model(**inputs)

        z = outputs.last_hidden_state                               # [B, 1+L, D]
        #pooled_z = outputs.pooler_output                           # [B, D]
        return z

class ImageEmbedder(nn.Module):
    """
    for image-to-video on cross-attn
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, num_patches, act_layer=nn.GELU(approximate='tanh')):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, 
                          out_features=hidden_size, act_layer=act_layer, drop=0)
        self.uncond_prob = uncond_prob
        assert self.uncond_prob >=0 and self.uncond_prob <= 1.0
        self.register_buffer("y_embedding", nn.Parameter(
            torch.randn(num_patches + 1, in_channels) / in_channels ** 0.5))
    
    def forward(self, embeddings, train, drop_ids=None):
        assert embeddings.ndim == 3          # [B, L, D]

        if train and (self.uncond_prob > 0):
            if drop_ids is None:
                drop_ids = torch.rand(embeddings.size(0)).cuda() < self.uncond_prob
            embeddings = torch.where(drop_ids[:, None, None], self.y_embedding, embeddings)
        
        embeddings = self.y_proj(embeddings)                                               
        return embeddings
    

class VideoEmbedder(nn.Module):
    """
    for image-to-video on cross-attn
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, num_patches, act_layer=nn.GELU(approximate='tanh')):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, 
                          out_features=hidden_size, act_layer=act_layer, drop=0)
        self.uncond_prob = uncond_prob
        assert self.uncond_prob >=0 and self.uncond_prob <= 1.0
        self.register_buffer("y_embedding", nn.Parameter(
            torch.randn(num_patches + 1, in_channels) / in_channels ** 0.5))
    
    def forward(self, embeddings, train, drop_ids=None):
        if not isinstance(embeddings, list):
            assert embeddings.ndim == 3          # [B, L, D]
            embeddings = [e for e in embeddings]

        if train and (self.uncond_prob > 0):
            if drop_ids is None:
                drop_ids = torch.rand(len(embeddings)).cuda() < self.uncond_prob
            #embeddings = torch.where(drop_ids[:, None, None], self.y_embedding, embeddings)
            embeddings = [e if not drop else self.y_embedding for drop, e in zip(drop_ids, embeddings)]

        embeddings = [self.y_proj(e) for e in embeddings]                                             
        return embeddings
    