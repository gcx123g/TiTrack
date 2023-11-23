# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Decoder for SeqTrack, modified from DETR transformer class.
"""

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from torch import Tensor
import torch.nn as nn
from lib.utils.pos_embed import get_sinusoid_encoding_table


class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_position_embeddings, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_embeds = self.word_embeddings(x)
        embeddings = input_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class SeqTrackDecoder(nn.Module):

    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, bins=1000, num_frames=9):
        super().__init__()
        self.num_frames = num_frames

        self.patch_embed = PatchEmbed(img_size=d_model, patch_size=16, in_chans=3, embed_dim=d_model)

        self.sa_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.cx_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.cy_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.cw_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.ch_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        self.num = 256
        self.pos_embed_src = nn.Parameter(torch.zeros(1, self.num, d_model))
        pos_embed_src = get_sinusoid_encoding_table(self.num, self.pos_embed_src.shape[-1], cls_token=False)
        self.pos_embed_src.data.copy_(torch.from_numpy(pos_embed_src).float().unsqueeze(0))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num, d_model))
        pos_embed = get_sinusoid_encoding_table(self.num, self.pos_embed.shape[-1], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.num + 1, d_model))
        pos_embed1 = get_sinusoid_encoding_table(self.num + 1, self.pos_embed1.shape[-1], cls_token=False)
        self.pos_embed1.data.copy_(torch.from_numpy(pos_embed1).float().unsqueeze(0))

        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.num + 2, d_model))
        pos_embed2 = get_sinusoid_encoding_table(self.num + 2, self.pos_embed2.shape[-1], cls_token=False)
        self.pos_embed2.data.copy_(torch.from_numpy(pos_embed2).float().unsqueeze(0))

        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.num + 3, d_model))
        pos_embed3 = get_sinusoid_encoding_table(self.num + 3, self.pos_embed3.shape[-1], cls_token=False)
        self.pos_embed3.data.copy_(torch.from_numpy(pos_embed3).float().unsqueeze(0))

        self.cx_embeddings = DecoderEmbeddings(2, d_model, 10, dropout)
        self.cy_embeddings = DecoderEmbeddings(2, d_model, 10, dropout)
        self.w_embeddings = DecoderEmbeddings(2, d_model, 10, dropout)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, seq):
        # flatten NxCxHxW to HWxNxC
        b = src.shape[0]
        src = src.permute(1, 0, 2)
        pos_embed_src = self.pos_embed_src.permute(1, 0, 2)
        seq = self.patch_embed(seq).permute(1, 0, 2)
        pos_embed = self.pos_embed.permute(1, 0, 2)

        # two self_attention for seq; two cross_attention for tgt memory
        for i in range(2):
            seq = self.sa_layer(seq, seq, pos_embed, pos_embed)

        # seq to coord
        src = self.cx_layer(tgt=src, memory=seq, pos=pos_embed, query_pos=pos_embed_src)
        img = src.permute(1, 0, 2)
        value, cx_tmp = torch.topk(img.softmax(-2).to(img), 1, -2)
        value, cx_tmp = torch.topk(value.squeeze(-2), 1, -1)
        cx = cx_tmp
        cx_tmp = (cx_tmp + 1) / 256
        cx_seq = self.cx_embeddings(cx_tmp.long()).permute(1, 0, 2)
        seq = torch.cat((seq, cx_seq), 0)
        pos_embed = self.pos_embed1.permute(1, 0, 2)

        src = self.cy_layer(tgt=src, memory=seq, pos=pos_embed, query_pos=pos_embed_src)
        img = src.permute(1, 0, 2)
        value, cy_tmp = torch.topk(img.softmax(-1).to(img), 1, -1)
        value, cy_tmp = torch.topk(value.squeeze(-1), 1, -1)
        cy = cy_tmp
        cy_tmp = (cy_tmp + 1) / 256
        cy_seq = self.cy_embeddings(cy_tmp.long()).permute(1, 0, 2)
        seq = torch.cat((seq, cy_seq), 0)
        pos_embed = self.pos_embed2.permute(1, 0, 2)

        src = self.cw_layer(tgt=src, memory=seq, pos=pos_embed, query_pos=pos_embed_src)
        img = src.permute(1, 0, 2)
        value, cy_tmp = torch.topk(img.softmax(-2).to(img), 1, -2)
        value, cy_tmp = torch.topk(value.squeeze(-2), 1, -1)
        cw = value * 256
        w = value
        w_seq = self.w_embeddings(w.long()).permute(1, 0, 2)
        seq = torch.cat((seq, w_seq), 0)
        pos_embed = self.pos_embed3.permute(1, 0, 2)

        src = self.ch_layer(tgt=src, memory=seq, pos=pos_embed, query_pos=pos_embed_src)
        img = src.permute(1, 0, 2)
        value, cy_tmp = torch.topk(img.softmax(-1).to(img), 1, -1)
        value, cy_tmp = torch.topk(value.squeeze(-1), 1, -1)
        ch = value * 256

        out = torch.cat((cx, cy, cw, ch), -1)
        return out


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_decoder(cfg):
    return SeqTrackDecoder(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.DECODER.DROPOUT,
        nhead=cfg.MODEL.DECODER.NHEADS,
        dim_feedforward=cfg.MODEL.DECODER.DIM_FEEDFORWARD,
        num_decoder_layers=cfg.MODEL.DECODER.DEC_LAYERS,
        normalize_before=cfg.MODEL.DECODER.PRE_NORM,
        return_intermediate_dec=False,
        bins=cfg.MODEL.BINS,
        num_frames=cfg.DATA.SEARCH.NUMBER
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
