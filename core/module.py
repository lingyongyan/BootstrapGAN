# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_softmax
from allennlp.nn import util as allen_util


class FlattenScaledDotProduct(nn.Module):
    def __init__(self, temperature, dropout=0.):
        super(FlattenScaledDotProduct, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, index=None, size=None):
        batchlize = q.dim() > 2
        if batchlize:
            attn_score = torch.einsum('bij, bij->bi', q, k)
        else:
            attn_score = torch.einsum('ij, ij->i', q, k)
        attn_score = attn_score / self.temperature
        attn_score = self.dropout(scatter_softmax(attn_score, index, dim=-1))
        return attn_score


class ScaledDotProduct(nn.Module):
    def __init__(self, temperature, dropout=0.):
        super(ScaledDotProduct, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None, size=None):
        batchlize = True if q.dim() > 2 else False

        if batchlize:
            attn_score = torch.bmm(q, k.transpose(-1, -2))
        else:
            attn_score = torch.mm(q, k.transpose(-1, -2))

        attn_score = attn_score / self.temperature
        attn_score = self.dropout(allen_util.masked_softmax(attn_score, mask, dim=-1))
        return attn_score


class FlattenAdditiveMul(nn.Module):
    def __init__(self, n_head, d_head, dropout=0.):
        super(FlattenAdditiveMul, self).__init__()
        self.attn = nn.Parameter(torch.Tensor(1, n_head, 2 * d_head))
        self.dropout = nn.Dropout(dropout)
        self.reset()

    def reset(self):
        glorot(self.attn)

    def forward(self, q, k, index=None, size=None):
        batchlize = True if q.dim() > 3 else False
        attn = self.attn
        if batchlize:
            attn = self.attn.unsqueeze(0)
        inp = torch.cat([q, k], dim=-1)
        attn_score = F.relu((inp*attn).sum(dim=-1))
        attn_score = self.dropout(scatter_softmax(attn_score, index, dim=1))
        return attn_score


class AdditiveMul(nn.Module):
    def __init__(self, n_head, d_head, dropout=0.):
        super(AdditiveMul, self).__init__()
        self.n_head, self.d_head = n_head, d_head
        self.attn = nn.Parameter(torch.Tensor(1, n_head, 2 * d_head))
        self.dropout = nn.Dropout(dropout)
        self.reset()

    def reset(self):
        glorot(self.attn)

    def forward(self, q, k, mask=None):
        batchlize = True if q.dim() > 3 else False
        attn = self.attn
        if batchlize:
            attn = attn.unsqueeze(0)
            n_q, n_k = q.size(1), k.size(1)
            q = q.unsqueeze(2).expand(1, 1, n_k, 1, 1)
            k = k.unsqueeze(1).expand(1, n_q, 1, 1, 1)
            inp = torch.cat([q, k], dim=-1)
            inp = inp.view(-1, n_q * n_k, self.n_head, self.d_head)
        else:
            n_q, n_k = q.size(0), k.size(0)
            q = q.unsqueeze(1).expand(1, n_k, 1, 1)
            k = k.unsqueeze(0).expand(n_q, 1, 1, 1)
            inp = torch.cat([q, k], dim=-1)
            inp = inp.view(n_q * n_k, self.n_head, self.d_head)
        attn_score = F.relu((inp*attn).sum(dim=-1))
        attn_score = self.dropout(allen_util.masked_softmax(attn_score, mask, dim=-1))
        if batchlize:
            attn_score = attn_score.view(-1, n_q, n_k, self.n_head)
        else:
            attn_score = attn_score.view(n_q, n_k, self.n_head)
        return attn_score
