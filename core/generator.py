# coding=utf-8

import torch
import torch.nn as nn
import numpy as np

from .model import GBNEncoder, GBNDecoder


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder = GBNEncoder(opt)
        self.decoder = GBNDecoder(opt)
        self.gamma = min(max(0., opt['gamma']), 1.0)

    def forward(self, graph_data, inps, n_iter=10):
        edge_index = graph_data.node_edge_index
        es, _ = self.encoder(graph_data)
        return self.decoder.expand(es, edge_index, inps, n_iter)

    def sample(self, graph_data, inps, n_iter=10,
               last_sample=10, sample_group=1, is_all_sample=None):
        edge_index = graph_data.node_edge_index
        es, _ = self.encoder(graph_data)
        logits, expansions, _ = self.decoder.expand(es, edge_index, inps,
                                                    n_iter,
                                                    last_sample=last_sample,
                                                    sample_group=sample_group,
                                                    is_all_sample=is_all_sample)
        return logits, expansions, es

    def get_PGLoss(self, outs, expansions, rewards):
        prev_rewards, rewards = rewards

        loss = 0
        for i, (iter_outs, iter_exps) in enumerate(zip(outs[-1], expansions[-1])):
            for j, (out, expansion) in enumerate(zip(iter_outs, iter_exps)):
                n = out.size(0)
                for k in range(n):
                    loss += - out[k][j] * rewards[i][j][k] / n
        loss = loss * np.power(self.gamma, len(expansions)-1) / len(expansions[-1])

        loss2 = 0
        for i, (iter_outs, iter_exps) in enumerate(zip(outs[:-1], expansions[:-1])):
            base = np.power(self.gamma, i)
            for j, (out, expansion) in enumerate(zip(iter_outs, iter_exps)):
                n = out.size(0)
                for k in range(n):
                    loss2 += - out[k][j] * prev_rewards[i][j][k] * base / n
        n = len(expansions)
        if self.gamma != 1.0:
            total = (1 - np.power(self.gamma, n)) / (1 - self.gamma)
        else:
            total = n
        return (loss + loss2) / total
