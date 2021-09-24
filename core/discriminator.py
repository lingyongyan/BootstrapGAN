# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LNClassifier, GBNEncoder
from .util import HLoss, calc_entropy


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        n_layer = opt['n_layer']
        opt['n_layer'] = 1
        self.encoder = GBNEncoder(opt)
        opt['n_layer'] = n_layer
        self.n_class = opt['n_class']
        self._lambda = 1.0

        # we set it an n-class classifier
        self.classifier = LNClassifier(opt['layer_dim'], self.n_class)

    def forward(self, graph_data, inp):
        es, ps = self._graph_encode(graph_data)
        return self._output(es, inp)
        # return self._output(inp)

    def _graph_encode(self, graph_data):
        es, ps = self.encoder(graph_data)
        return es, ps

    def _output(self, es, inp):
        if inp is None:
            raise ValueError('inputs should be a tensor')
        inp = es[inp]
        if inp is not None:
            logit = self.classifier(inp)
        else:
            logit = None
        return logit

    def batch_classify(self, graph_data, inps):
        es, ps = self._graph_encode(graph_data)
        rewards = []
        base = 1.0 / self.n_class
        for n_step, step_inps in enumerate(inps):
            step_rewards = []
            for cate, cate_inps in enumerate(step_inps):
                cate_logits = self._output(es, cate_inps)
                cate_probs = F.softmax(cate_logits, dim=-1)
                reward = cate_probs[:, cate] - base
                step_rewards.append(reward)
            rewards.append(step_rewards)
        return rewards

    def get_loss(self, graph_data, inps, targets):
        real_inps, fake_inps = inps
        assert real_inps.size(0) == targets.size(0)
        es, ps = self._graph_encode(graph_data)
        real_logits = self._output(es, real_inps)
        fake_logits = self._output(es, fake_inps)
        criterion1 = HLoss()
        criterion2 = nn.CrossEntropyLoss()
        loss1_1 = criterion1(real_logits)  # entropy of real input (minimize)
        loss1_2 = -criterion1(fake_logits)  # entropy of fake input (maximize)
        loss2 = criterion2(real_logits, targets)  # cross entropy loss
        loss = loss1_2 + self._lambda * (loss1_1 + loss2)
        return loss
