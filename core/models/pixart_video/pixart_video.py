#coding=utf-8
import math
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
#import xformers.ops
from einops import rearrange, repeat
import torch.nn.functional as F

from core.models.utils import auto_grad_checkpoint, get_layernorm, set_grad_checkpoint
from core.models.pos_embed import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed, PositionEmbedding2D, LlamaRotaryEmbedding
from core.models.embedder import CaptionEmbedder, TimestepEmbedder, LabelEmbedder, PatchEmbed3D, ImageEmbedder
from core.models.attention import Attention, CrossAttention, LlamaMLP, AttentionRoPE, broadcat, FinalLayer

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
        if self.channel_first:
            shape = [1] * x.ndim
            shape[1] = x.size(1)
        else:
            shape = [x.size(-1)]
        x = F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma.view(shape) 
        if self.bias is not None:
            x = x + self.bias.view(shape)
        return x

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class Block(nn.Module):
    """
    A PixArtVideo block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, 
                 hidden_size, 
                 num_heads, 
                 mlp_ratio=4.0, 
                 drop_path=0.,
                 mlp_type="",
                 #enable_rope=False,
                 enable_rope_spatial=False,
                 enable_rope_temporal=False,
                 use_fused_layernorm=False,
                 temp_window_size=None,
                 norm_type="",
                 qk_norm=False,
                 image_jump_temporal=False,
                 causal_temporal_block=False,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        if isinstance(temp_window_size, str):
            temp_window_size = eval(temp_window_size)
        self.temp_window_size = temp_window_size
        self.image_jump_temporal = image_jump_temporal
        self.causal_temporal_block = causal_temporal_block

        self.enable_rope_spatial = enable_rope_spatial
        if self.enable_rope_spatial:
            attn_func_spatial = AttentionRoPE
        else:
            attn_func_spatial = Attention

        self.enable_rope_temporal = enable_rope_temporal
        if self.enable_rope_temporal:
            attn_func_temporal = AttentionRoPE
        else:
            attn_func_temporal = Attention

        assert norm_type  in ["layernorm", "rmsnorm", "llamarmsnorm"]
        if norm_type == 'llamarmsnorm':
            self.norm1 = LlamaRMSNorm(hidden_size)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(hidden_size, channel_first=False)
        else:
            self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_fused=use_fused_layernorm)

        # spatial attn
        self.attn = attn_func_spatial(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
        self.cross_attn = CrossAttention(hidden_size, num_heads, qk_norm=qk_norm)

        if norm_type == 'llamarmsnorm':
            self.norm2 = LlamaRMSNorm(hidden_size)
        elif norm_type == 'rmsnorm':
            self.norm2 = RMSNorm(hidden_size, channel_first=False)
        else:
            self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_fused=use_fused_layernorm)
        
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if mlp_type == 'timm':
            self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        elif mlp_type == 'llama':
            self.mlp = LlamaMLP(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

        self.attn_temp = attn_func_temporal(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)

    def causal_mask(self, wt, wh, ww, dtype, device):
        # causal along T dimention only
        mask_cond = torch.zeros((wt, wh, ww), dtype=dtype, device=device)
        for i in range(wt):
            mask_cond[i, ...] += i
        mask_cond = mask_cond.reshape(-1)
        mask_cond = mask_cond < (mask_cond + 1).view(-1, 1)     # (wt*wh*ww, wt*wh*ww)

        mask = torch.full((wt*wh*ww, wt*wh*ww), torch.finfo(dtype).min, device=device)
        mask.masked_fill_(mask_cond, 0)
        return mask.unsqueeze(0)
    
    def forward(self, x, y, t, y_mask=None, tpe=None, T=1, H=1, W=1, freqs_cos=None, freqs_sin=None, y_im=None, y_im_mask=None):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        
        # spatial attn
        x_s = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x_s = rearrange(x_s, "B (T S) C -> (B T) S C", T=T, S=H*W)
        if self.enable_rope_spatial:
            freqs_cos_s = rearrange(freqs_cos, "B T H W C -> (B T) (H W) C")
            freqs_sin_s = rearrange(freqs_sin, "B T H W C -> (B T) (H W) C")
            x_s = self.attn(x_s, freqs_cos_s, freqs_sin_s)
        else:
            x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T, S=H*W)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal branch
        if T == 1:
            pass
        else:
            x_t = rearrange(x, "B (T S) C -> (B S) T C", T=T, S=H*W)
            if tpe is not None:
                x_t = x_t + tpe

            wt, wh, ww = self.temp_window_size
            wt = wt if T > 1 else 1                 # case of image
            if wt == -1:
                wt = T                              # global attention for temporal
            nt, nh, nw = T // wt, H // wh, W // ww
            x_t = rearrange(x_t, "(B H W) T C -> B T H W C", H=H, W=W)
            x_t = rearrange(x_t, "B (nt wt) (nh wh) (nw ww) C -> (B nt nh nw) (wt wh ww) C", wt=wt, wh=wh, ww=ww)
            
            if self.causal_temporal_block:
                causal_mask = self.causal_mask(wt, wh, ww, x_t.dtype, x_t.device)
            else:
                causal_mask = None
        
            if self.enable_rope_temporal:
                freqs_cos_t = rearrange(freqs_cos, "B (nt wt) (nh wh) (nw ww) C -> (B nt nh nw) (wt wh ww) C", wt=wt, wh=wh, ww=ww)
                freqs_sin_t = rearrange(freqs_sin, "B (nt wt) (nh wh) (nw ww) C -> (B nt nh nw) (wt wh ww) C", wt=wt, wh=wh, ww=ww)
                x_t = self.attn_temp(x_t, freqs_cos_t, freqs_sin_t, mask=causal_mask)
            else:
                x_t = self.attn_temp(x_t, mask=causal_mask)
            x_t = rearrange(x_t, "(B nt nh nw) (wt wh ww) C -> B (nt wt nh wh nw ww) C", wt=wt, wh=wh, ww=ww, nt=nt, nh=nh, nw=nw)
            x = x + self.drop_path(gate_msa * x_t)

        # cross attn
        if y is not None:
            x = x + self.cross_attn(x, y, y_mask)               # text condition in attn mode
        if y_im is not None:            
            x = x + self.cross_attn(x, y_im, y_im_mask)         # image condition in attn mode

        # mlp
        x_mlp = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.drop_path(gate_mlp * self.mlp(x_mlp))
        return x

class PixArtVideo(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(self, 
                 input_size=(1, 32, 32), 
                 patch_size=(1, 2, 2), 
                 in_channels=4, 
                 hidden_size=1152, 
                 depth=28, 
                 num_heads=16, 
                 mlp_ratio=4.0, 
                 class_dropout_prob=0.1, 
                 image_dropout_prob=0.1,
                 pred_sigma=True, 
                 drop_path: float = 0., 
                 caption_channels=4096, 
                 model_max_length=120,
                 space_scale=1.0,
                 time_scale=1.0,
                 mlp_type="",
                 norm_type="layernorm",
                 prob_self_condition=0.,
                 prob_img_condition=0.,
                 img_condition_type="concat",
                 img_condition_w_mask=False,
                 img_condition_augment=False,
                 prob_img_condition_attn=0.,
                 img_condition_attn_channels=1024,
                 img_condition_attn_num_patches=256,
                 prob_text_condition=1.0,
                 #enable_rope=False,
                 use_fused_layernorm=False,
                 temp_window_size=None,
                 adain_with_text=False,
                 qk_norm=False,
                 grad_checkpointing=False,
                 image_jump_temporal=False,

                 enable_frames_embedder=False,
                 enable_src_size_embedder=False,
                 enable_tgt_size_embedder=False,
                 enable_crop_size_embedder=False,

                 position_embed_spaltial="rope",
                 position_embed_temporal="rope",

                 causal_temporal_block=False,

                 cfg_text=True,
                 cfg_self_cond=True,
                 cfg_image_cond=False,
                 cfg_image_cond_attn=True,
                 ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.prob_self_condition = prob_self_condition                  # if probability > 0, then enable self-condition
        self.prob_img_condition = prob_img_condition                    # if probability > 0, then enable img-condition on concat mode
        self.prob_img_condition_attn = prob_img_condition_attn          # if probability > 0, then enable img-condition on cross-attn mode
        self.prob_text_condition = prob_text_condition                  # if probability > 0, then enable text-condition on cross-attn mode
        self.adain_with_text = adain_with_text
        self.img_condition_type = img_condition_type
        assert self.img_condition_type in ['concat', 'frame-prediction']
        self.img_condition_w_mask = img_condition_w_mask

        self.grad_checkpointing = grad_checkpointing
        self.image_jump_temporal = image_jump_temporal

        self.image_dropout_prob = image_dropout_prob

        if prob_img_condition > 0:
            if self.img_condition_type == 'concat':
                #self.in_channels = in_channels * (1 + int(prob_self_condition > 0) + int(prob_img_condition > 0))

                self.in_channels = in_channels * (1 + int(prob_self_condition > 0))
                self.imgcond_adater = nn.Sequential(
                    nn.Linear(in_channels * 2, in_channels * 8, bias=True),
                    nn.SiLU(),
                    nn.Linear(in_channels * 8, in_channels, bias=True),
                )

            elif self.img_condition_type == 'frame-prediction':
                self.in_channels = in_channels * (1 + int(prob_self_condition > 0)) + int(self.img_condition_w_mask)
            else:
                raise ValueError()
        else:
            self.in_channels = in_channels * (1 + int(prob_self_condition > 0))


        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.space_scale = space_scale
        self.time_scale = time_scale
        
        self.img_condition_augment = img_condition_augment
        if self.img_condition_augment:
            from core.models.utils_im_cond import TransformerV2
            concat_dim = 8
            adapter_transformer_layers = 1

            self.local_image_concat = nn.Sequential(
                nn.Conv2d(in_channels, concat_dim * 4, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=1, padding=1))
            
            self.local_temporal_encoder = TransformerV2(
                    heads=2, dim=concat_dim, dim_head_k=concat_dim, dim_head_v=concat_dim, 
                    dropout_atte = 0.05, mlp_dim=concat_dim, dropout_ffn = 0.05, depth=adapter_transformer_layers)

        self.x_embedder = PatchEmbed3D(patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.enable_frames_embedder = enable_frames_embedder
        if self.enable_frames_embedder:
            self.frame_embedder = TimestepEmbedder(hidden_size)

        self.enable_src_size_embedder = enable_src_size_embedder
        if self.enable_src_size_embedder:
            self.src_size_embedder = TimestepEmbedder(hidden_size // 2)
        self.enable_tgt_size_embedder = enable_tgt_size_embedder
        if self.enable_tgt_size_embedder:
            self.tgt_size_embedder = TimestepEmbedder(hidden_size // 2)
        self.enable_crop_size_embedder = enable_crop_size_embedder
        if self.enable_crop_size_embedder:
            self.crop_size_embedder = TimestepEmbedder(hidden_size // 4)

        #self.enable_rope = enable_rope
        self.position_embed_spaltial = position_embed_spaltial
        self.position_embed_temporal = position_embed_temporal
        assert self.position_embed_spaltial in ['rope', 'absolute']
        assert self.position_embed_temporal in ['rope', 'absolute'] 

        # if self.position_embed_spaltial == 'rope' or self.position_embed_temporal == 'rope':
        #     self.enable_rope = True 
        # else:
        #     self.enable_rope = False
        
        if self.position_embed_spaltial == 'absolute' and self.position_embed_temporal == 'absolute':
            self.register_buffer("pos_embed", self.get_spatial_pos_embed())
            self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        elif self.position_embed_spaltial == 'rope' and self.position_embed_temporal == 'rope':
            self.head_dim = hidden_size // num_heads
            # freqs_cos, freqs_sin = self.compute_freqs(32, 16, 16, self.head_dim // 3)
            # self.register_buffer("freqs_cos", freqs_cos)
            # self.register_buffer("freqs_sin", freqs_sin)
            self.enable_rope_spatial = True 
            self.enable_rope_temporal = True

        elif self.position_embed_spaltial == 'absolute' and self.position_embed_temporal == 'rope':
            self.pos_embed = PositionEmbedding2D(hidden_size)
            self.pos_embed_temporal_rope = LlamaRotaryEmbedding(hidden_size // num_heads, max_position_embeddings=1000)
            self.enable_rope_spatial = False 
            self.enable_rope_temporal = True

        else:
            raise ValueError()

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        if self.prob_text_condition > 0:
            self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, 
                                            act_layer=approx_gelu, token_num=model_max_length)

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(hidden_size, 
                  num_heads, 
                  mlp_ratio=mlp_ratio, 
                  drop_path=drop_path[i],
                  mlp_type=mlp_type,
                  #enable_rope=self.enable_rope,
                  enable_rope_spatial=self.enable_rope_spatial,
                  enable_rope_temporal=self.enable_rope_temporal,
                  use_fused_layernorm=use_fused_layernorm,
                  temp_window_size=temp_window_size,
                  norm_type=norm_type,
                  qk_norm=qk_norm,
                  image_jump_temporal=image_jump_temporal,
                  causal_temporal_block=causal_temporal_block)
            for i in range(depth)
        ])
        assert self.patch_size[0] == 1  # for both image and video
        self.final_layer = FinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

        self.initialize_weights()

        if self.grad_checkpointing:
            for blk in self.blocks:
                set_grad_checkpoint(blk)
        
        # config for cfg inference
        self.cfg_text = cfg_text
        self.cfg_image_cond = cfg_image_cond
        self.cfg_self_cond = cfg_self_cond
        self.cfg_image_cond_attn = cfg_image_cond_attn

    def compute_freqs(self, t, h, w, dim, max_freq=10, 
                      scaling_factor_t=1.0, scaling_factor_h=1.0, scaling_factor_w=1.0):
        freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi                            # [dim//2]

        t_t = torch.arange(t) / scaling_factor_t                                                                   # [t]
        h_t = torch.arange(h) / scaling_factor_h                                                                  # [h]                                   
        w_t = torch.arange(w) / scaling_factor_w                                                                  # [w]

        t_freqs = torch.einsum('..., f -> ... f', t_t, freqs)                                   # [t, dim//2]
        h_freqs = torch.einsum('..., f -> ... f', h_t, freqs)                                   # [h, dim//2]
        w_freqs = torch.einsum('..., f -> ... f', w_t, freqs)                                   # [w, dim//2]
        
        t_freqs = repeat(t_freqs, '... n -> ... (n r)', r=2)                                    # [t, dim]
        h_freqs = repeat(h_freqs, '... n -> ... (n r)', r=2)                                    # [h, dim]
        w_freqs = repeat(w_freqs, '... n -> ... (n r)', r=2)                                    # [w, dim]

        freqs = broadcat((t_freqs[:, None, None, :], 
                          h_freqs[None, :, None, :], 
                          w_freqs[None, None, :, :]), dim=-1)                                   # [t, h, w, 3*dim]

        freqs_cos = freqs.cos().unsqueeze(0)                                                    # [1, t, h, w, 3*dim]
        freqs_sin = freqs.sin().unsqueeze(0)                                                    # [1, t, h, w, 3*dim]

        return freqs_cos, freqs_sin

    def forward(self, x, timestep, y, y_mask=None, x_self_cond=None, x_img_cond=None, y_im=None, size_infos=[]):
        """
        Forward pass of PixArtVideo.
        x: (N, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        y_im: (N, L, C)
        """
        assert x.ndim == 5
        is_video = x.size(2) > 1
        batch, c, f, h, w = x.shape
        num_frames = f
        T_p, H_p, W_p = self.patch_size
        T_input, H_input, W_input = x.shape[-3:]

        x = x.to(self.dtype)
        x_zeros = torch.zeros_like(x).detach()

        drop_ids_img_cond = None
            
        timestep = timestep.to(self.dtype)

        # embedding
        x, T, H, W = self.x_embedder(x)

        #if self.enable_rope:
        if self.position_embed_spaltial == 'rope' and self.position_embed_temporal == 'rope':
            B = len(x)

            freqs_cos, freqs_sin = self.compute_freqs(T, H, W, self.head_dim // 3,
                                    #scaling_factor_t=max(T, 32) / 32.0, 
                                    scaling_factor_t=1.0, 
                                    scaling_factor_h=H / 16.0, 
                                    scaling_factor_w=W / 16.0)
            freqs_cos = freqs_cos.to(x).repeat(B, 1, 1, 1, 1)
            freqs_sin = freqs_sin.to(x).repeat(B, 1, 1, 1, 1)

        elif self.position_embed_spaltial == 'absolute' and self.position_embed_temporal == 'rope':
            scale = ((H_input * 8) * (W_input * 8)) ** 0.5 / 512
            base_size = round((H_input * W_input / H_p / W_p) ** 0.5)
            pos_embed = self.pos_embed(x, H_input//H_p, W_input//W_p, scale=scale, base_size=base_size)     # [1, S, C]

            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=H*W)
            x = x + pos_embed
            x = rearrange(x, "B T S C -> B (T S) C")

            freqs_cos, freqs_sin = self.pos_embed_temporal_rope(x, seq_len=T)                               # [T, C]
            freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(len(x), 1, H, W, 1)              # [b, t, h, w, c]
            freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(len(x), 1, H, W, 1) 
        else:
            raise ValueError()
            

        if (self.prob_text_condition > 0) and (y is not None):
            y = y.to(self.dtype)
            y = self.y_embedder(y, self.training)           # (B, 1, L, D)

            if y_mask is not None:
                if y_mask.shape[0] != y.shape[0]:
                    y_mask = y_mask.repeat(y.shape[0] // y_mask.shape[0], 1)
                y_mask = y_mask.squeeze(1).squeeze(1)
                y = y.squeeze(1).masked_select(y_mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
                y_lens = y_mask.sum(dim=1).tolist()
            else:
                y_lens = [y.shape[2]] * y.shape[0]
                y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            y = y_lens = None
        
        t = self.t_embedder(timestep.to(x.dtype))       # [B, C]
        if self.adain_with_text and y is not None:
            y_pooled = torch.stack([_.mean(dim=0) for _ in torch.split(y.squeeze(0), y_lens, dim=0)], dim=0)        # [batch, C]
            t = t + y_pooled

        if self.enable_frames_embedder:
            frame_embedding = self.frame_embedder(torch.tensor([num_frames] * len(x)).to(x.device).to(x.dtype))
            t = t + frame_embedding

        t0 = self.t_block(t)                            # [B, C]
        
        if is_video and (self.prob_img_condition_attn > 0) and (y_im is not None):
            y_im = y_im.to(self.dtype)
            y_im = self.y_im_embedder(y_im, self.training, drop_ids_img_cond)      # [B, L, D]
            y_im_lens = [y_im.size(1)] * y_im.size(0)
            y_im = rearrange(y_im, "B L D -> 1 (B L) D")
        else:
            y_im = y_im_lens = None

        for i, block in enumerate(self.blocks):
            if not self.enable_rope_temporal:
                if i == 0:
                    tpe = self.pos_embed_temporal
                else:
                    tpe = None
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, tpe, 
                                        T, H, W, None, None, y_im, y_im_lens)                                 # (B, L, D) #support grad checkpoint
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, None, 
                                        T, H, W, freqs_cos, freqs_sin, y_im, y_im_lens) 

        x = self.final_layer(x, t)  # (N, L, Tp * Hp * Wp * out_channels)
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=T,
            N_h=H,
            N_w=W,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )

        # remove the prediction corresponding to the condition image/video
        if is_video and self.prob_img_condition > 0 and (self.img_condition_type == 'frame-prediction'):
            x = x[:, :, x_img_cond.size(2):, :, :]
        return x

    def forward_with_dpmsolver(self, x, timestep, y, y_mask=None, x_self_cond=None, x_img_cond=None, y_im=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, y_mask, x_self_cond=x_self_cond, x_img_cond=x_img_cond, y_im=y_im)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, y_mask=None, x_self_cond=None, x_img_cond=None, y_im=None, **kwargs):
        """
        Forward pass of PixArtVideo, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        if y is not None:
            # make sure y is the concat of [y, null_y] outside, in 'scripts/sample_batch/sample_batch.py'
            if not self.cfg_text:
                y_half = y[:len(y) // 2]
                y = torch.cat([y_half, y_half], dim=0)    

        if x_self_cond is not None:
            if self.cfg_self_cond:
                x_self_cond_half = x_self_cond[:len(x_self_cond) // 2]
                x_self_cond = torch.cat([x_self_cond_half, torch.zeros_like(x_self_cond_half)], dim=0)
        
        # if x_img_cond is not None:
        #     if self.cfg_image_cond:
        #         x_img_cond_half = x_img_cond[:len(x_img_cond) // 2]
        #         x_img_cond = torch.cat([x_img_cond_half, torch.zeros_like(x_img_cond_half)], dim=0)
        
        if y_im is not None:
            # make sure y_im is the concat of [y_im, null_y_im] outside, in 'scripts/sample_batch/sample_batch.py'
            if not self.cfg_image_cond_attn:
                y_im_half = y_im[:len(y_im) // 2]
                y_im = torch.cat([y_im_half, y_im_half], dim=0)

        model_out = self.forward(combined, timestep, y, y_mask=y_mask, 
                                 x_self_cond=x_self_cond, x_img_cond=x_img_cond, y_im=y_im, size_infos=kwargs.get('size_infos', None))         # [batch, self.out_channels, t, h, w]
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        
        if True:
            eps, rest = model_out[:, :3], model_out[:, 3:]  #TODO: why only apply cfg to first three channels
        else:
            if self.pred_sigma:
                eps, rest = model_out[:, :self.out_channels//2], model_out[:, self.out_channels//2:]
            else:
                eps, rest = model_out, model_out[:, 0:0]
                
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.enable_frames_embedder:
            # zero init
            nn.init.normal_(self.frame_embedder.mlp[0].weight, std=0.02)
            nn.init.constant_(self.frame_embedder.mlp[2].weight, 0)
            nn.init.constant_(self.frame_embedder.mlp[2].bias, 0)


        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        if self.prob_text_condition > 0:
            nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        if self.prob_img_condition_attn > 0:
            nn.init.normal_(self.y_im_embedder.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_im_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArtVideo blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        
        # Zero-out temporal blocks
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

def PixArtVideo_XL_1x2x2(from_pretrained="", **kwargs):
    model = PixArtVideo(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)

    if os.path.isfile(from_pretrained):
        assert 'pixart' in from_pretrained.lower()
        ckpt = torch.load(from_pretrained)
        state_dict = ckpt['state_dict']

        state_dict["x_embedder.proj.weight"] = state_dict["x_embedder.proj.weight"].unsqueeze(2)
        del state_dict["pos_embed"]

        if model.blocks[0].scale_shift_table.shape != state_dict['blocks.0.scale_shift_table'].shape:
            key_list = list(state_dict.keys())
            for k in key_list:
                if 'scale_shift_table' in k:
                    del state_dict[k]
                    print(f"remove weight of {k}")
        
        if model.t_block[1].weight.shape != state_dict['t_block.1.weight'].shape:
            key_list = list(state_dict.keys())
            for k in key_list:
                if 't_block' in k:
                    del state_dict[k]
                    print(f"remove weight of {k}")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print(f"Loading from {from_pretrained}")
    
    return model

def PixArtVideo_XL_1x1x1(from_pretrained="", **kwargs):
    model = PixArtVideo(depth=28, hidden_size=1152, patch_size=(1, 1, 1), num_heads=16, **kwargs)

    if os.path.isfile(from_pretrained):
        assert 'pixart' in from_pretrained.lower()
        ckpt = torch.load(from_pretrained)
        state_dict = ckpt['state_dict']

        state_dict["x_embedder.proj.weight"] = state_dict["x_embedder.proj.weight"].unsqueeze(2)
        del state_dict["pos_embed"]

        if model.blocks[0].scale_shift_table.shape != state_dict['blocks.0.scale_shift_table'].shape:
            key_list = list(state_dict.keys())
            for k in key_list:
                if 'scale_shift_table' in k:
                    del state_dict[k]
                    print(f"remove weight of {k}")
        
        if model.t_block[1].weight.shape != state_dict['t_block.1.weight'].shape:
            key_list = list(state_dict.keys())
            for k in key_list:
                if 't_block' in k:
                    del state_dict[k]
                    print(f"remove weight of {k}")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print(f"Loading from {from_pretrained}")
    
    return model


def PixArtVideo_g_1x2x2(from_pretrained="", **kwargs):
    model = PixArtVideo(depth=40, hidden_size=1536, patch_size=(1, 2, 2), num_heads=16, **kwargs)

    if os.path.isfile(from_pretrained):
        assert 'pixart' in from_pretrained.lower()
        ckpt = torch.load(from_pretrained)
        state_dict = ckpt['state_dict']

        state_dict["x_embedder.proj.weight"] = state_dict["x_embedder.proj.weight"].unsqueeze(2)
        del state_dict["pos_embed"]

        if model.blocks[0].scale_shift_table.shape != state_dict['blocks.0.scale_shift_table'].shape:
            key_list = list(state_dict.keys())
            for k in key_list:
                if 'scale_shift_table' in k:
                    del state_dict[k]
                    print(f"remove weight of {k}")
        
        if model.t_block[1].weight.shape != state_dict['t_block.1.weight'].shape:
            key_list = list(state_dict.keys())
            for k in key_list:
                if 't_block' in k:
                    del state_dict[k]
                    print(f"remove weight of {k}")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print(f"Loading from {from_pretrained}")
    
    return model

def PixArtVideo_dummy_1x2x2(from_pretrained="", **kwargs):
    model = PixArtVideo(depth=4, hidden_size=576, patch_size=(1, 2, 2), num_heads=8, **kwargs)

    if os.path.isfile(from_pretrained):
        assert 'pixart' in from_pretrained.lower()
        ckpt = torch.load(from_pretrained)
        state_dict = ckpt['state_dict']

        state_dict["x_embedder.proj.weight"] = state_dict["x_embedder.proj.weight"].unsqueeze(2)
        del state_dict["pos_embed"]

        if model.blocks[0].scale_shift_table.shape != state_dict['blocks.0.scale_shift_table'].shape:
            key_list = list(state_dict.keys())
            for k in key_list:
                if 'scale_shift_table' in k:
                    del state_dict[k]
                    print(f"remove weight of {k}")
        
        if model.t_block[1].weight.shape != state_dict['t_block.1.weight'].shape:
            key_list = list(state_dict.keys())
            for k in key_list:
                if 't_block' in k:
                    del state_dict[k]
                    print(f"remove weight of {k}")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print(f"Loading from {from_pretrained}")
    
    return model