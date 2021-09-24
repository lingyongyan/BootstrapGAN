# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import GRUCell
from allennlp.nn import util as allen_util

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from .sub_layer import AttentionLayer, EmbeddingEncodingLayer
from .sub_layer import CombineLayer, MatchingLayer

n_depth = 2 + 1
n_edge = 3 + 1

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)


class GNNConv(MessagePassing):
    def __init__(self, d_in, d_attr, dropout=0.0, eps=0, d_out=None,
                 train_eps=True, bias=True, global_sighted=True, concat=True,
                 aggr='add', flow='source_to_target'):
        super(GNNConv, self).__init__(aggr=aggr, flow=flow)
        n_head = 1
        if d_out is None:
            d_out = d_in
        d_head = d_out // n_head
        self.bias = bias
        self.global_sighted = global_sighted
        if self.global_sighted:
            self.depth_encoder = EmbeddingEncodingLayer(n_depth, d_attr)
            self.edge_encoder = EmbeddingEncodingLayer(n_edge, d_attr)
        else:
            self.edge_encoder = None
            self.depth_encoder = None
        d_k = d_in + 2 * d_attr if self.global_sighted else d_in
        self.attn_layer = AttentionLayer(d_in, d_k, d_in,
                                         n_head, d_head, d_head,
                                         concat=concat, bias=bias,
                                         attn_method='flatten_sdp',
                                         dropout=dropout)

        self.w_res = nn.Linear(d_in, d_out, bias=bias)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset()

    def reset(self):
        self.eps.data.fill_(self.initial_eps)
        glorot(self.w_res.weight)
        if self.bias:
            zeros(self.w_res.bias)

    def forward(self, x, edge_index, edge_attr):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        size = (x[j].size(0), x[i].size(0))
        x = (x[j], x[i])
        out = self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)
        return out

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        q, v = x_i, x_j
        if self.global_sighted:
            depth_embedding = self.depth_encoder(edge_attr[:, 0])
            edge_embedding = self.edge_encoder(edge_attr[:, 1])
            k = torch.cat([x_j, depth_embedding, edge_embedding], dim=-1)
        else:
            k = x_j
        out, _ = self.attn_layer(q, k, v, index=edge_index_i, size=size_i)
        return out

    def update(self, aggr_out, x):
        res = self.w_res(x[1])
        out = (1 + self.eps) * res + aggr_out
        return out


class InputLayer(nn.Module):
    def __init__(self, dim, n_class):
        super(InputLayer, self).__init__()
        self.dim = dim
        self.combine_layer = CombineLayer(dim, dim//2)

    def forward(self, hx, inp, mask=None):
        if hx is None:
            inp = allen_util.masked_mean(inp, mask.unsqueeze(-1).bool(), dim=1)
        else:
            inp = self.combine_layer(hx, inp, inp, mask)
        return inp


class MemoryLayer(nn.Module):
    def __init__(self, dim):
        super(MemoryLayer, self).__init__()
        self.memory_cell = GRUCell(dim, dim)

    def forward(self, hx, inp):
        hx = self.memory_cell(inp, hx)
        return hx


class OutputLayer(nn.Module):
    def __init__(self, dim):
        super(OutputLayer, self).__init__()
        self.score_layer = MatchingLayer(dim, dim)

    def forward(self, hs, es, mask, n_output, stochastic=None):
        expansions, expansion_scores = [], []

        original_scores = self.score_layer(hs, es)
        mu, std = original_scores.mean(), original_scores.std()
        scores = (original_scores - mu) / std  # normalize

        cate_logits = F.log_softmax(scores, dim=0)
        sample_logits = F.log_softmax(scores, dim=-1)
        logits = sample_logits + cate_logits

        d_scores = scores.detach()
        # row probs used to sample top entities for each category
        row_probs = allen_util.masked_softmax(d_scores, mask=mask, dim=-1)
        row_probs = row_probs.clamp(min=1e-6)

        # col probs used to sample one category to each entity
        col_probs = allen_util.masked_softmax(d_scores, mask=mask, dim=0)
        col_probs = col_probs.clamp(min=1e-6)

        if stochastic is None:
            stochastic = self.training

        if stochastic:
            index = torch.multinomial(col_probs.t(), 1).view(-1)
            for i in range(scores.size(0)):
                mask_i = mask[i, :].bool() & (index == i)
                candidates = torch.nonzero(mask_i, as_tuple=False)
                n_sample = min(n_output, candidates.size(0))
                if n_sample > 0:
                    top = torch.multinomial(row_probs[i][mask_i], n_sample)
                    top = top.view(-1)
                else:
                    top = []
                expansion = candidates[top].view(-1)
                expansions.append(expansion)
                expansion_scores.append(logits[:, expansion].t())
        else:
            index = col_probs.argmax(dim=0)
            for i in range(scores.size(0)):
                mask_i = mask[i, :].bool() & (index == i)
                candidates = torch.nonzero(mask_i, as_tuple=False)
                top = torch.argsort(row_probs[i][mask_i], descending=True)
                top = top[:n_output]
                expansion = candidates[top].view(-1)
                expansions.append(expansion)
                expansion_scores.append(logits[:, expansion].t())
        return expansions, expansion_scores

