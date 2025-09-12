from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple
from functools import partial

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils import to_2tuple
from .pos_embed import get_2d_sincos_pos_embed


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model, n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor, model_type: str = 'ClearCLIP', ignore_residual=True, output_cls_token=False, last_n_layers=1):
        B, nc, w, h = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        for blk in self.transformer.resblocks[:-last_n_layers]:
            x = blk(x)

        output = 0
        for blk in self.transformer.resblocks[-last_n_layers:]:
            if ignore_residual:
                output += self.custom_attn(blk.attn, blk.ln_1(x), model_type=model_type)

                x = blk(x)
            else:
                x_out = x + self.custom_attn(blk.attn, blk.ln_1(x), model_type=model_type)
                x_out = x_out + blk.mlp(blk.ln_2(x_out))
                output += x_out

                x = blk(x)

        x = output.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj
            tokens = tokens @ self.proj

        if output_cls_token:
            return pooled, tokens
        
        return tokens

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size[0]
        h0 = h // self.patch_size[1]
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    @staticmethod
    def gaussian_window(dim1, dim2, std=1.):
        constant = 1 / (std * math.sqrt(2))
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant,
                               end=(start + (dim - 1)) * constant,
                               steps=dim,
                               dtype=torch.float)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing='ij')) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, adjust_for_cls=True):
        m = torch.einsum('ij,kl->ijkl', torch.eye(dim1), torch.eye(dim2))
        m = m.permute((0, 3, 1, 2)).contiguous()  # m[ijkl] = 1 iff (i, j) == (k, l)
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
        out = out.view(dim1 * dim2, dim1 * dim2)
        if adjust_for_cls:
            v_adjusted = torch.vstack([torch.zeros((1, dim1 * dim2)), out])
            out = torch.hstack([torch.zeros((dim1 * dim2 + 1, 1)), v_adjusted])
        return out

    @staticmethod
    def create_spatial_kernel(height, width, std=1.0, kernel_type='gaussian'):
        """
        Args:
            height, width: 核的维度
            std: 标准差，控制空间衰减速度
            kernel_type: 核类型选项
                - 'gaussian': 高斯核，基于L2距离平方
                - 'laplacian_l1': 拉普拉斯核，基于L1距离（曼哈顿距离）
        Returns:
            torch.Tensor: 空间核
        数学公式：
            高斯核: K(x,y) = exp(-((x-u)²+（y-u）²)/(2σ²))
            拉普拉斯核(L1): K(x,y) = exp(-(|x-u|+|y-u|)/σ)
        """
        # 计算中心点坐标
        center_h, center_w = (height - 1) / 2.0, (width - 1) / 2.0
        
        # 创建坐标网格
        y_coords = torch.arange(height, dtype=torch.float) - center_h
        x_coords = torch.arange(width, dtype=torch.float) - center_w
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        if kernel_type == 'gaussian':
            # 高斯核: K(x,y) = exp(-((x-u)²+（y-u）²)/(2σ²))
            distance_squared = (x_grid**2 + y_grid**2) / (2 * std**2)
            return torch.exp(-distance_squared)
        elif kernel_type == 'laplacian':
            # 拉普拉斯核(L1距离)：K(x,y) = exp(-(|x-u|+|y-u|)/σ)
            distance_l1 = torch.abs(x_grid) + torch.abs(y_grid)
            return torch.exp(-distance_l1 / std)
        else:
            raise ValueError(f"不支持的核类型: {kernel_type}")
    
    @staticmethod 
    def build_positional_attention_bias(patch_h, patch_w, spatial_kernel, adjust_for_cls=True):
        """
        使相邻patch获得更高的注意力权重
        Args:
            patch_h, patch_w: patch网格的高度和宽度
            spatial_kernel: 空间核
            adjust_for_cls: 是否为CLS token预留位置
        Returns:
            torch.Tensor: 局部位置增强
        """
        total_patches = patch_h * patch_w
        identity_tensor = torch.eye(total_patches).view(total_patches, patch_h, patch_w)
        convolved = F.conv2d(
            identity_tensor.unsqueeze(1),
            spatial_kernel.unsqueeze(0).unsqueeze(1),
            padding='same'
        ).squeeze(1)
        attention_matrix = convolved.view(total_patches, total_patches)
        
        # 如果需要调整CLS token
        if adjust_for_cls:
            # 为CLS token添加零填充
            cls_adjusted = torch.zeros((total_patches + 1, total_patches + 1))
            cls_adjusted[1:, 1:] = attention_matrix
            return cls_adjusted
            
        return attention_matrix
    
    def custom_attn(self, attn_layer, x, model_type='ClearCLIP'):

        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if model_type == 'vanilla':
            qk_attn = torch.bmm(q, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(qk_attn, dim=-1)
        elif model_type == 'MaskCLIP':
            mask = torch.empty(q.shape[1], q.shape[1], dtype=q.dtype).to(q.device)
            mask.fill_(float('-inf'))
            mask.fill_diagonal_(0)
            mask = mask.unsqueeze(0).repeat(q.shape[0], 1, 1)
            attn_weights = F.softmax(mask, dim=-1)
        elif model_type == 'SCLIP':
            qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale
            kk_attn = torch.bmm(k, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(qq_attn, dim=-1) + F.softmax(kk_attn, dim=-1)
        elif model_type == 'SegEarth':
            qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale
            kk_attn = torch.bmm(k, k.transpose(1, 2)) * scale
            vv_attn = torch.bmm(v, v.transpose(1, 2)) * scale
            attn_weights = F.softmax(qq_attn, dim=-1) + F.softmax(kk_attn, dim=-1) + F.softmax(vv_attn, dim=-1)
        elif model_type == 'ClearCLIP':
            qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale
            attn_weights = F.softmax(qq_attn, dim=-1)
        elif model_type == 'RSCLIP':
            # 配置参数
            self.gaussian_std = 1.0
            self.use_mixed_kernels = True  # 启用混合核函数
            if not hasattr(self, 'positional_bias_cache'):
                self.positional_bias_cache = {}
            # 计算patch维度
            patches_per_side = int(np.sqrt(num_tokens - 1))
            patch_shape = (patches_per_side, patches_per_side)
            # 检查缓存 - 添加混合核标识
            cache_key = (patch_shape, self.use_mixed_kernels)
            if cache_key not in self.positional_bias_cache:
                # 计算高斯窗口大小
                window_height = patches_per_side * 2 - 1
                window_width = patches_per_side * 2 - 1
                # 生成混合空间感知核：组合高斯核和拉普拉斯核
                if self.use_mixed_kernels:
                    gaussian_kernel = self.create_spatial_kernel(
                        window_height, window_width, std=self.gaussian_std, kernel_type='gaussian'
                    )
                    laplacian_kernel = self.create_spatial_kernel(
                        window_height, window_width, std=self.gaussian_std, kernel_type='laplacian'
                    )
                    spatial_kernel =  0.7*gaussian_kernel +  0.3*laplacian_kernel
                else:
                    # 纯高斯核
                    spatial_kernel = self.create_spatial_kernel(
                        window_height, window_width, std=self.gaussian_std, kernel_type='gaussian'
                    )
                # 构建位置注意力偏置矩阵
                positional_bias = self.build_positional_attention_bias(
                    patches_per_side, patches_per_side,
                    spatial_kernel, adjust_for_cls=True
                )
                # 转换数据类型并缓存
                self.positional_bias_cache[cache_key] = positional_bias.unsqueeze(0).to(
                    dtype=x.dtype, device=x.device
                )
            # 获取缓存的位置偏置矩阵
            positional_bias = self.positional_bias_cache[cache_key]

            q_norm = q.norm(dim=2, keepdim=True)
            k_norm = k.norm(dim=2, keepdim=True)
            norm_product = torch.matmul(q_norm, k_norm.transpose(-2, -1))
            # 位置偏置 × 缩放因子 × 范数乘积
            omega = positional_bias.clone() * scale * norm_product.detach()

            # 重塑张量以处理每个注意力头
            q_heads = q.view(bsz, num_heads, -1, head_dim)
            k_heads = k.view(bsz, num_heads, -1, head_dim)
            v_heads = v.view(bsz, num_heads, -1, head_dim)
            # 初始化头部输出张量
            head_outputs = torch.zeros_like(q_heads)
            # 对不同的注意力头应用不同的膨胀率
            for h in range(num_heads):
                # 使用线性膨胀率
                dilation_rate = h + 1  # 线性膨胀率: 1, 2, 3, 4...
                # dilation_rate = 2 ** (h % 3)  # 得到的膨胀率为 1, 2, 4, 1, 2, 4, ...
                # 为当前头应用膨胀注意力
                if dilation_rate == 1:
                    # 膨胀率为1时的标准注意力
                    q_h = q_heads[:, h]
                    k_h = k_heads[:, h]
                    v_h = v_heads[:, h]
                    qq_attn = torch.bmm(q_h, q_h.transpose(1, 2)) * scale
                    kk_attn = torch.bmm(k_h, k_h.transpose(1, 2)) * scale
                    vv_attn = torch.bmm(v_h, v_h.transpose(1, 2)) * scale
                    attn_h_q = F.softmax(qq_attn, dim=-1)
                    attn_h_k = F.softmax(kk_attn, dim=-1)
                    attn_h_v = F.softmax(vv_attn, dim=-1)
                    head_outputs[:, h] = torch.bmm((attn_h_q+attn_h_k+attn_h_v), v_h)
                else:
                    # 对当前头的键和值进行子采样
                    q_dilated = q_heads[:, h, ::dilation_rate]
                    k_dilated = k_heads[:, h, ::dilation_rate]
                    v_dilated = v_heads[:, h, ::dilation_rate]  # 对值也进行相同的子采样
                    q_h = q_heads[:, h]
                    k_h = k_heads[:, h]
                    v_h = v_heads[:, h]
                    qq_attn = torch.bmm(q_h, q_dilated.transpose(1, 2)) * scale
                    kk_attn = torch.bmm(k_h, k_dilated.transpose(1, 2)) * scale
                    vv_attn = torch.bmm(v_h, v_dilated.transpose(1, 2)) * scale
                    attn_h_q = F.softmax(qq_attn, dim=-1)
                    attn_k_q = F.softmax(kk_attn, dim=-1)
                    attn_v_q = F.softmax(vv_attn, dim=-1)

                    dilated_output = torch.bmm((attn_h_q+attn_k_q+attn_v_q), v_dilated)
                    head_outputs[:, h] = dilated_output
            # 重塑头部输出以匹配原始格式
            MMA = 0.5 * head_outputs.view(bsz * num_heads, -1, head_dim).transpose(0, 1).contiguous()

            qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale
            kk_attn = torch.bmm(k, k.transpose(1, 2)) * scale
            vv_attn = torch.bmm(v, v.transpose(1, 2)) * scale
            self_attn = F.softmax(qq_attn, dim=-1) + F.softmax(kk_attn, dim=-1) + F.softmax(vv_attn, dim=-1)

            qk_attn = torch.bmm(q, k.transpose(1, 2)) * scale
            qv_attn = torch.bmm(q, v.transpose(1, 2)) * scale
            kv_attn = torch.bmm(k, v.transpose(1, 2)) * scale
            # 自适应权重
            alpha = torch.sigmoid(torch.mean(qk_attn))
            beta = torch.sigmoid(torch.mean(qv_attn))
            gamma = torch.sigmoid(torch.mean(kv_attn))
            cross_attn = alpha * F.softmax(qk_attn, dim=-1) + beta * F.softmax(qv_attn, dim=-1) + gamma * F.softmax(
                kv_attn, dim=-1)
            SCE_and_NAP = torch.bmm((1.0*self_attn+0.5*cross_attn+0.5*F.softmax(omega, dim=-1)), v)
            # 输出
            attn_output = SCE_and_NAP.transpose(0, 1)+ MMA

        elif model_type in ['NACLIP', 'NOnly', 'GAV']:
            self.gaussian_std = 1.0 # 5.0
            self.addition_cache = dict()
            n_patches = (int(np.sqrt((num_tokens - 1))), int(np.sqrt((num_tokens - 1))))
            addition = self.addition_cache.get(n_patches)
            if addition is None:
                window_size = [side * 2 - 1 for side in n_patches]
                window = VisionTransformer.gaussian_window(*window_size, std=self.gaussian_std)
                addition = VisionTransformer.get_attention_addition(*n_patches, window).unsqueeze(0).to(x.dtype).to(x.device)
                self.addition_cache[n_patches] = addition
            omega = addition.clone()

            if model_type == 'NACLIP':
                attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale
            elif model_type == 'NOnly':
                attn_weights = torch.zeros((num_heads, num_tokens, num_tokens)).to(x.dtype).to(x.device)
                omega = omega * scale * torch.einsum('hop,hPO->hpP', q.norm(dim=2).unsqueeze(1), k.norm(dim=2).unsqueeze(2)).detach()
            elif model_type == 'GAV':
                attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale
                omega = omega * scale * torch.einsum('hop,hPO->hpP', q.norm(dim=2).unsqueeze(1), k.norm(dim=2).unsqueeze(2)).detach()
            else:
                raise NotImplemented
            attn_weights += omega
            attn_weights = F.softmax(attn_weights, dim=-1)
        if model_type != 'RSCLIP':
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        else:
            attn_output = attn_output.contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        return attn_output

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: int = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):

        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
