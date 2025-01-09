# This file includes code from the <ViT> project
# Original source code can be found at <https://github.com/FrancescoSaverioZuppichini/ViT>

# Code obtained from <https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/transfomer.md>
# Author: <FrancescoSaverioZuppichini>

# Code modifications:
# <Adapted skeleton data>

import math
import warnings
import pdb
from functools import partial

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary
from visualizer import get_local

from . import models
from .drop import DropPath
from .utils import trunc_normal_
# get_local.activate()

    
class JointEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, temporal_segment_size: int = 4,
                 spatio_size: int = 25, temporal_size: int = 120, emb_size: int = 256):
        self.temporal_segment_size = temporal_segment_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('B N M T V C -> (B N M) C T V'),
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=(temporal_segment_size, 1),
                      stride=(temporal_segment_size, 1)),
            # Rearrange('N C T V -> N (T V) C'),
        )
        # zeros or randn?
        # self.spatio_pos_emb = nn.Parameter(torch.randn(1, emb_size, 1, spatio_size))
        # self.temporal_pos_emb = nn.Parameter(torch.randn(1, emb_size,
        #                                                  temporal_size // temporal_segment_size, 1))
        self.spatio_pos_emb = nn.Parameter(torch.zeros(1, emb_size, 1, spatio_size))
        self.temporal_pos_emb = nn.Parameter(torch.zeros(1, emb_size,
                                                         temporal_size // temporal_segment_size, 1))
        # self.mask_token_emb = nn.Parameter(torch.zeros(1, 1, emb_size))
        trunc_normal_(self.spatio_pos_emb, std=.02)
        trunc_normal_(self.spatio_pos_emb, std=.02)
        # trunc_normal_(self.mask_token_emb, std=.02)
                
    def forward(self, x: Tensor, bool_masked_pos = None) -> Tensor:
        x = self.projection(x)
        # batch_size, _, temporal_size, spatio_size = x.shape
        spatio_pos_emb = self.spatio_pos_emb.expand_as(x)
        temporal_pos_emb = self.temporal_pos_emb.expand_as(x)
        spatio_pos_emb = rearrange(spatio_pos_emb, 'b e t v -> b (t v) e')
        temporal_pos_emb = rearrange(temporal_pos_emb, 'b e t v -> b (t v) e')
        x = rearrange(x, 'b e t v -> b (t v) e')
        # if bool_masked_pos is not None:
        #     # replace mask token
        #     mask_token_emb = self.mask_token_emb.expand_as(x)
        #     # w = rearrange(bool_masked_pos, 'b m t v 1 -> (b m) (t v) 1').type_as(mask_token_emb)
        #     w = bool_masked_pos.type_as(mask_token_emb)
        #     x = x * (1 - w) + mask_token_emb * w
        # add pos embedding
        x = x + spatio_pos_emb + temporal_pos_emb
        # x = rearrange(x, 'N C T V -> N (T V) C')
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768,
                 num_heads: int = 8,
                 att_drop: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(att_drop)
        self.proj = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    @get_local('att')
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        att = F.softmax(energy * self.scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_p)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# class FeedForwardBlock(nn.Sequential):
#     def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
#         super().__init__(
#             nn.Linear(emb_size, expansion * emb_size),
#             nn.GELU(),
#             nn.Dropout(drop_p),
#             nn.Linear(expansion * emb_size, emb_size),
#         )


# class TransformerEncoderBlock(nn.Sequential):
#     def __init__(self,
#                  emb_size: int = 256,
#                  att_drop_p: float = 0.,
#                  forward_expansion: int = 4,
#                  forward_drop_p: float = 0.,
#                  drop_path_p: float = 0.,
#                  ** kwargs):
#         super().__init__(
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 MultiHeadAttention(emb_size, att_drop=att_drop_p, **kwargs),
#                 nn.Dropout(forward_drop_p),
#                 DropPath(drop_path_p)
#             )),
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 FeedForwardBlock(
#                     emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
#                 nn.Dropout(forward_drop_p),
#                 DropPath(drop_path_p)
#             )
#             ))


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 256,
                 att_drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 drop_path_p: float = 0.,
                 norm_layer = nn.LayerNorm,
                 layer_scale_init_value: float = None,
                 ** kwargs):
        super().__init__()
        self.norm1 = norm_layer(emb_size)
        self.attn = MultiHeadAttention(emb_size, att_drop=att_drop_p, **kwargs)
        self.norm2 = norm_layer(emb_size)
        self.mlp = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.drop = nn.Dropout(forward_drop_p)
        self.drop_path = DropPath(drop_path_p)

        if layer_scale_init_value is not None and layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((emb_size)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((emb_size)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.drop(self.attn(self.norm1(x))))
            x = x + self.drop_path(self.drop(self.mlp(self.norm2(x))))
        else:
            x = x + self.drop_path(self.gamma_1 * self.drop(self.attn(self.norm1(x))))
            x = x + self.drop_path(self.gamma_2 * self.drop(self.mlp(self.norm2(x))))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 8, drop_path_p: float = 0., **kwargs):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_p, depth)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                drop_path_p=dpr[i], **kwargs)
             for i in range(depth)])

    def forward(self, x, output_hidden_states: bool = True):
        all_hidden_states = (x,) if output_hidden_states else None
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
        
        outputs = (x,) + all_hidden_states if output_hidden_states else (x,)
        return outputs
                

@models.register('ClassificationHeadLight')
class ClassificationHeadLight(nn.Sequential):
    def __init__(self, emb_size: int = 256, n_classes: int = 120,
                 num_persons: int = 2, drop_p: float = 0.):
        super().__init__(
				#     nn.LayerNorm(emb_size), 
            Rearrange('(b m) n e -> b m n e', m=num_persons),
            Reduce('b m n e -> b e', reduction='mean'),
            nn.BatchNorm1d(emb_size, affine=False),
            nn.Dropout(drop_p),
            nn.Linear(emb_size, n_classes))


@models.register('ClassificationHeadLarge')
class ClassificationHeadLarge(nn.Sequential):
    def __init__(self, emb_size: int = 256,
                 n_classes: int = 120,
                 hidden_dim: int = 2048,
                 num_persons: int = 2,
                 num_joints: int = 25,
                 drop_p: float = 0.):
        super().__init__(
            # nn.LayerNorm(emb_size), 
            nn.Dropout(drop_p),
            Rearrange('(B M) (T J) C -> B M T J C', M=num_persons, J=num_joints),
            Reduce('B M T J C -> B M J C', reduction='mean'),
            Rearrange('B M J C -> B M (J C)'),
            Reduce('B M C -> B C', reduction='mean'),
            nn.Linear(emb_size*num_joints, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(drop_p),
            nn.Linear(hidden_dim, n_classes)
        )


@models.register('ClassificationHeadTest')
class ClassificationHeadTest(nn.Module):
    def __init__(self, emb_size: int = 256,
                 n_classes: int = 120,
                 hidden_dim: int = 2048,
                 num_persons: int = 2,
                 num_joints: int = 25,
                 drop_p: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_persons = num_persons
        self.num_joints = num_joints
        self.drop_p = drop_p
        self.drop = nn.Dropout(drop_p)
        self.bn = nn.BatchNorm1d(emb_size)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(emb_size*4, hidden_dim)
        self.cls_head = nn.Linear(emb_size, n_classes)
    
    def forward(self, x):
        x = rearrange(x, '(b m) (t v) c -> b m t v c',
                      m=self.num_persons, v=self.num_joints)
        x = reduce(x, 'b m t v c -> b c', reduction='mean')
        # x = self.drop(x)
        # x = self.fc(x)
        x = self.bn(x)
        # x = self.relu(x)
        x = self.drop(x)
        x = self.cls_head(x)
        return x
                 
@models.register('SkT')
class SkT(nn.Module):
    def __init__(self,     
                in_channels: int = 3,
                temporal_segment_size: int = 4,
                spatio_size: int = 25,
                temporal_size: int = 120,
                emb_size: int = 256,
                depth: int = 12,
                num_heads: int = 8,
                att_drop_p: float = 0.,
                forward_drop_p: float = 0.,
                drop_path_p: float = 0.,
                norm_layer = partial(nn.LayerNorm, eps=1e-6),
                layer_scale_init_value: float = None,
                **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.embedding = JointEmbedding(in_channels, temporal_segment_size,
                        spatio_size, temporal_size, emb_size)
        self.encoder = TransformerEncoder(depth,
                                          drop_path_p,
                                          emb_size=emb_size,
                                          num_heads=num_heads,
                                          att_drop_p=att_drop_p,
                                          forward_drop_p=forward_drop_p,
                                          norm_layer=norm_layer,
                                          layer_scale_init_value=layer_scale_init_value,
                                          **kwargs)
    
    def forward(self, x: torch.Tensor, output_hidden_states: bool = True, bool_masked_pos = None):
        x = self.embedding(x, bool_masked_pos)
        encoder_ouputs = self.encoder(x, output_hidden_states=output_hidden_states)
        return {
            'last_hidden_state': encoder_ouputs[0],
            'hidden_states': encoder_ouputs[1:] if output_hidden_states else None,
        }


@models.register('SkTWithDecoder')
class SkTWithDecoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 temporal_segment_size: int = 4,
                 spatio_size: int = 25,
                 temporal_size: int = 120,
                 encoder_emb_size: int = 256,
                 decoder_emb_size: int = 256,
                 encoder_depth: int = 8,
                 decoder_depth: int = 4,
                 num_heads: int = 8,
                 att_drop_p: float = 0.,
                 forward_drop_p: float = 0.,
                 drop_path_p: float = 0.,
                 norm_layer = partial(nn.LayerNorm, eps=1e-6),
                 layer_scale_init_value: float = None,
                 mask_strategy: str = 'random',
                 using_feat_head: bool = True,
                 using_motion_head: bool = False,
                 **kwargs):
        super().__init__()
        self.spatio_size = spatio_size
        self.temporal_size = temporal_size
        self.temporal_segment_size = temporal_segment_size
        self.temporal_segments = temporal_size // temporal_segment_size
        self.encoder_emb_size = encoder_emb_size
        self.using_motion_head = using_motion_head
        self.using_feat_head = using_feat_head
        assert mask_strategy in ['random', 'tube', 'motion'],\
            f'Unknown mask strategy: {mask_strategy}'
        self.mask_strategy = mask_strategy
        # encoder
        self.encoder_embedding = JointEmbedding(in_channels, temporal_segment_size,
                        spatio_size, temporal_size, encoder_emb_size)
        self.encoder = TransformerEncoder(encoder_depth,
                                          drop_path_p,
                                          emb_size=encoder_emb_size,
                                          num_heads=num_heads,
                                          att_drop_p=att_drop_p,
                                          forward_drop_p=forward_drop_p,
                                          norm_layer=norm_layer,
                                          layer_scale_init_value=layer_scale_init_value,
                                          **kwargs)
        self.encoder_norm = norm_layer(encoder_emb_size)
        # decoder
        self.decoder_emb_size = decoder_emb_size
        self.decoder_embedding = nn.Linear(encoder_emb_size, decoder_emb_size)
        self.decoder = TransformerEncoder(decoder_depth,
                                          drop_path_p,
                                          emb_size=decoder_emb_size,
                                          num_heads=num_heads,
                                          att_drop_p=att_drop_p,
                                          forward_drop_p=forward_drop_p,
                                          norm_layer=norm_layer,
                                          layer_scale_init_value=layer_scale_init_value,
                                          **kwargs)
        self.decoder_norm = norm_layer(decoder_emb_size)
        # decoder prediction head
        if using_feat_head:
            self.decoder_feat_head = nn.Linear(decoder_emb_size, encoder_emb_size)
        if using_motion_head:
            self.decoder_motion_head = nn.Linear(decoder_emb_size,
                                                temporal_segment_size*in_channels)

        self.decoder_spatio_pos_emb = nn.Parameter(torch.zeros(1, decoder_emb_size, 1, spatio_size))
        self.decoder_temporal_pos_emb = nn.Parameter(torch.zeros(1, decoder_emb_size,
                                                                 self.temporal_segments, 1))
        self.mask_token_emb = nn.Parameter(torch.zeros(1, 1, decoder_emb_size))
        trunc_normal_(self.decoder_spatio_pos_emb, std=.02)
        trunc_normal_(self.decoder_temporal_pos_emb, std=.02)
        trunc_normal_(self.mask_token_emb, std=.02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def tube_masking(self, x, mask_ratio, tube_len):
        N, L, D = x.shape  # batch, length, dim
        VP, TP = self.spatio_size, self.temporal_segments
        len_VP_keep = int(VP * (1 - mask_ratio))

        # Divide the dimension of time into several tubes
        assert TP % tube_len == 0
        TP_ = TP // tube_len
        noise = torch.rand(N, TP_, VP, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=-1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=-1)

        # Fill the tubes to restore the original dimension of time
        ids_shuffle = ids_shuffle.repeat_interleave(tube_len, dim=1)
        ids_restore = ids_restore.repeat_interleave(tube_len, dim=1)

        # shift tube
        # shift = torch.randint(low=0, high=ids_shuffle.size(1), size=(1,)).item()
        # ids_shuffle = torch.roll(ids_shuffle, shifts=shift, dims=1)
        # ids_restore = torch.roll(ids_restore, shifts=shift, dims=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_VP_keep]
        x_masked = torch.gather(x.view(N, TP, VP, D), dim=2,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))
        x_masked = rearrange(x_masked, 'n t v d -> n (t v) d')

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, TP, VP], device=x.device)
        mask[:, :, :len_VP_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)
        mask = rearrange(mask, 'n t v -> n (t v)')

        return x_masked, mask, ids_restore, ids_keep

    def motion_aware_tube_masking(self, x, x_motion, tube_len,
                                  mask_ratio: float = 0.9, s_tau: float = 0.2,
                                  t_tau: float = 0.2):
        # pass
        N, L, D = x.shape  # batch, length, dim
        VP, TP = self.spatio_size, self.temporal_segments
        len_VP_keep = int(VP * (1 - mask_ratio))
        s = self.temporal_segment_size

        assert TP % tube_len == 0
        TP_ = TP // tube_len

        x_motion = rearrange(x_motion, 'b n m (t s) v c -> (b n m) t v (s c)', s=s)
        motion_intensity = x_motion.abs()
        motion_intensity = reduce(motion_intensity, 'n t v c -> n t v', 'mean')
        motion_intensity = motion_intensity.view(-1, TP_, tube_len, VP).mean(dim=2)
        motion_intensity = motion_intensity / (motion_intensity.max(dim=2, keepdim=True).values + 1e-6)

        # Divide the dimension of time into several tubes
        noise = torch.rand(N, TP_, VP, device=x.device)  # noise in [0, 1]
        noise = noise + s_tau * motion_intensity

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=-1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=-1)

        # Fill the tubes to restore the original dimension of time
        ids_shuffle = ids_shuffle.repeat_interleave(tube_len, dim=1)
        ids_restore = ids_restore.repeat_interleave(tube_len, dim=1)

        # shift tube
        # shift = torch.randint(low=0, high=ids_shuffle.size(1), size=(1,)).item()
        # ids_shuffle = torch.roll(ids_shuffle, shifts=shift, dims=1)
        # ids_restore = torch.roll(ids_restore, shifts=shift, dims=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_VP_keep]
        x_masked = torch.gather(x.view(N, TP, VP, D), dim=2,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))
        x_masked = rearrange(x_masked, 'n t v d -> n (t v) d')

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, TP, VP], device=x.device)
        mask[:, :, :len_VP_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)
        mask = rearrange(mask, 'n t v -> n (t v)')

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio: float = 0.,
                        tube_len: int = 6,
                        x_motion: torch.Tensor = None,
                        s_tau: float = 0.2, t_tau: float = 0.2,
                        output_hidden_states=True):
        x = self.encoder_embedding(x, bool_masked_pos=None)
        if mask_ratio > 0.:
            if self.mask_strategy == 'random':
                x, mask, id_restore, _ = self.random_masking(x, mask_ratio)
            elif self.mask_strategy == 'tube':
                x, mask, id_restore, _ = self.tube_masking(x, mask_ratio, tube_len)
            elif self.mask_strategy == 'motion':
                x, mask, id_restore, _ = self.motion_aware_tube_masking(x, x_motion, tube_len, mask_ratio, s_tau, t_tau)
            else:
                raise NotImplementedError
        else:
            mask = None
            id_restore = None
        x = self.encoder(x, output_hidden_states=output_hidden_states)
        return {
            'last_hidden_state': self.encoder_norm(x[0]),
            'hidden_states': x[1:] if output_hidden_states else None,
            'mask': mask,
            'id_restore': id_restore
        }

    def forward_decoder(self, x, id_restore):
        NM = x.shape[0]
        TP = self.temporal_segments
        VP = self.spatio_size

        x = self.decoder_embedding(x)
        C = x.shape[-1]

        mask_tokens = self.mask_token_emb.repeat(NM, TP * VP - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        if self.mask_strategy == 'random':
            x_ = torch.gather(
                x_, dim=1, index=id_restore[:, :, None].repeat(1, 1, C)
            )  # unshuffle
            x = rearrange(x_, 'b (t v) c -> b c t v', t=TP, v=VP)
        elif self.mask_strategy in ['tube', 'motion']:
            x_ = rearrange(x_, 'b (t v) c -> b t v c', t=TP, v=VP)
            x_ = torch.gather(
                x_, dim=2, index=id_restore[:, :, :, None].repeat(1, 1, 1, C)
            )
            x = rearrange(x_, 'b t v c -> b c t v')
        else:
            raise NotImplementedError

        # add pos & temp embed
        x = x + self.decoder_spatio_pos_emb + self.decoder_temporal_pos_emb
        x = rearrange(x, 'b c t v -> b (t v) c')

        # apply Transfomer Decoder, only output last hidden states
        x = self.decoder(x, output_hidden_states=False)[0]
        x = self.decoder_norm(x)

        # predictor projection
        res = {}
        if self.using_feat_head:
            x_feat = self.decoder_feat_head(x)
            res['feat'] = x_feat
        if self.using_motion_head:
            x_motion = self.decoder_motion_head(x)
            res['motion'] = x_motion

        return res    

    def forward(self, x, mask_ratio: float = 0.8, tube_len: int = 6,
                x_motion: torch.Tensor = None, s_tau: float = 0.2, t_tau: float = 0.2):
        encoder_outputs = self.forward_encoder(x, mask_ratio, tube_len, x_motion, s_tau, t_tau)
        id_restore = encoder_outputs['id_restore']
        mask = encoder_outputs['mask']
        latent = encoder_outputs['last_hidden_state']
        pred = self.forward_decoder(latent, id_restore)

        return pred, mask


@models.register('SkTForClassification')
class SkTForClassification(nn.Module):
    def __init__(self,
                 encoder_spec: dict,
                 cls_head_spec: dict,
                 encoder_pretrain_weight: str = None,
                 encoder_freeze: bool = False,
                 drop_path_p: float = 0.0,
                 using_ema_model: bool = True,
                 init_scale: float = 0.001,
                 reinit_last_norm: bool = False,
                 reinit_last_n_layers: int = 0
                 ):
        super().__init__()
        self.init_scale = init_scale
        self.reinit_last_norm = reinit_last_norm
        self.reinit_last_n_layer = reinit_last_n_layers
        if encoder_pretrain_weight:
            sv_file = torch.load(encoder_pretrain_weight)
            sv_file['model']['args']['model_spec']['args']['drop_path_p'] = drop_path_p
            loaded_model = models.make(sv_file['model'], load_sd=True)
            if using_ema_model:
                self.encoder = loaded_model.ema.model
            else:
                self.encoder = loaded_model.auto_encoder
            if reinit_last_norm or reinit_last_n_layers > 0:
                self.encoder.encoder_norm = nn.LayerNorm(self.encoder.encoder_emb_size)
        else:
            self.encoder = models.make(encoder_spec)
        if encoder_freeze:
            self.encoder.requires_grad_(False)
        else:
            self.encoder.requires_grad_(True)
        self.encoder_freeze = encoder_freeze
        self.emb_size = self.encoder.encoder_emb_size
        # self.norm = nn.LayerNorm(self.emb_size*4)

        self.cls_head = models.make(cls_head_spec,
                                    args={'emb_size': self.emb_size})
        if encoder_pretrain_weight:
            # if have pretrain weight, only init cls head
            # self.cls_head.apply(self._xavier_init_weights)
            if encoder_freeze:
                self.cls_head.apply(self._linear_init_weights)
            else:
                self.cls_head.apply(self._finetune_init_weights)
                if reinit_last_n_layers > 0:
                    for i in range(reinit_last_n_layers):
                        self.encoder.encoder.layers[-1 - i].apply(self._finetune_init_weights)
        else:
            self.apply(self._finetune_init_weights)

    def _linear_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _finetune_init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            trunc_normal_(m.weight, std=.02)
            m.weight.data.mul_(self.init_scale)
            # torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                m.bias.data.mul_(self.init_scale)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return ('encoder.encoder_embedding.spatio_pos_emb',
                'encoder.encoder_embedding.temporal_pos_emb')

    def extract_feat(self, x):
        return self.encoder.forward_encoder(x)['last_hidden_state']
        # feats = self.encoder.forward_encoder(x)['hidden_states'][-4:]
        # feat = torch.concatenate(feats, dim=-1)
        # # feat = sum(feats) / len(feats)
        # return self.norm(feat)

    def forward_train(self, x):
        feat = self.extract_feat(x)
        return self.cls_head(feat)

    def forward_test(self, x):
        B, NC, M, T, V, C = x.shape
        feat = self.extract_feat(x)
        cls_score = self.cls_head(feat)
        cls_score = rearrange(cls_score, '(b nc) e -> b nc e', b=B, nc=NC)
        cls_score = cls_score.mean(dim=1)
        return cls_score

    def forward(self, x: torch.Tensor):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)


if __name__ == '__main__':
    summary(SkTForClassification(SkT()), (1, 2, 120, 25, 3), device='cpu')
