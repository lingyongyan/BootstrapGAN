# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import GNNConv, InputLayer, MemoryLayer, OutputLayer
from .sub_layer import MLPLayer
from .graph_util import get_cate_mask, get_cate_neighbors

n_depth = 2 + 1
n_edge = 3 + 1


class model_check(nn.Module):
    def save(self, optimizer, filename):
        params = {
            'model': self.state_dict(),
            'optim': optimizer.state_dict()
        }
        try:
            print('print model to path:%s' % filename)
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, optimizer, filename, device):
        try:
            print('load model from path:%s' % filename)
            checkpoint = torch.load(filename, map_location=device)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        return optimizer


class GBNEncoder(nn.Module):
    def __init__(self, opt, JK="last"):
        super(GBNEncoder, self).__init__()
        self.n_layer = opt['n_layer']
        self.d_feature = opt['feature_dim']
        self.d_layer = opt['layer_dim']
        self.d_edge = opt['edge_feature_dim']
        self.dropout = opt['dropout']
        self.JK = JK

        if self.n_layer < 1:
            raise ValueError("Number of GNN layers must be greater than 0.")

        # List of GNNLayers
        self.node_nns = nn.ModuleList()
        self.edge_nns = nn.ModuleList()
        for layer in range(self.n_layer):
            if layer == 0:
                d_in = self.d_feature
            else:
                d_in = self.d_layer
            d_out = self.d_layer
            self.node_nns.append(GNNConv(d_in, self.d_edge,
                                         dropout=opt['dropout'],
                                         bias=opt['bias'],
                                         d_out=d_out,
                                         global_sighted=not opt['local'],
                                         flow='target_to_source'))
            self.edge_nns.append(GNNConv(d_in, self.d_edge,
                                         dropout=opt['dropout'],
                                         bias=opt['bias'],
                                         d_out=d_out,
                                         global_sighted=not opt['local'],
                                         flow='source_to_target'))

        self.norms = torch.nn.ModuleList()
        for layer in range(self.n_layer):
            self.norms.append(nn.BatchNorm1d(self.d_layer))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        h_list = [x]
        for layer in range(self.n_layer):
            h_i = self.node_nns[layer](h_list[layer], edge_index, edge_attr)
            h_j = self.edge_nns[layer](h_list[layer], edge_index, edge_attr)

            h_i = F.relu(h_i)
            h_j = F.relu(h_j)

            h = torch.cat([h_i, h_j], dim=0)
            h = self.norms[layer](h)
            h_i, h_j = h[:h_i.size(0)], h[h_i.size(0):]

            h_i = F.dropout(h_i, self.dropout, training=self.training)
            h_j = F.dropout(h_j, self.dropout, training=self.training)
            h_list.append([h_i, h_j])

        # Different implementations of Jk-concat
        if self.JK == "concat":
            h_0 = torch.cat([h[0] for h in h_list[1:]], dim=1)
            h_1 = torch.cat([h[1] for h in h_list[1:]], dim=1)
            output = (h_0, h_1)
        elif self.JK == "last":
            output = h_list[-1]
        elif self.JK == "max":
            h0 = torch.stack([h[0] for h in h_list[1:]], dim=0)
            h0 = torch.max(h0, dim=0)[0]
            h1 = torch.stack([h[1] for h in h_list[1:]], dim=0)
            h1 = torch.max(h1, dim=0)[0]
            output = (h0, h1)
        elif self.JK == "sum":
            h0 = torch.stack([h[0] for h in h_list[1:]], dim=0)
            h0 = torch.sum(h0, dim=0)
            h1 = torch.stack([h[1] for h in h_list[1:]], dim=0)
            h1 = torch.sum(h1, dim=0)
            output = (h0, h1)

        return output


class GBNDecoder(nn.Module):
    def __init__(self, opt):
        super(GBNDecoder, self).__init__()
        self.input_layer = InputLayer(opt['layer_dim'], opt['n_class'])
        self.memory_layer = MemoryLayer(opt['layer_dim'])
        self.output_layer = OutputLayer(opt['layer_dim'])
        self.min_match = opt['min_match']
        self.n_expansion = opt['n_expansion']
        self.sim_metric = opt['sim_metric']

    def expand(self, es, edge_index, seeds, n_iter,
               last_sample=-1, sample_group=-1, is_all_sample=None):
        outputs = []
        expansions = []
        hses = []
        if isinstance(seeds, list):
            n_class = len(seeds)
        else:
            n_class = 1
            seeds = [seeds]
        seed_mask = torch.zeros(es.size(0)).to(es.device, torch.bool)
        seed_mask.scatter_(0, torch.cat(seeds, dim=0), 1)
        last_expansion = seeds
        cate_masks = get_cate_mask(seeds, es.size(0))
        hs = None
        for i in range(n_iter):
            if torch.sum((seed_mask == 0)).float() == 0:
                break
            hs = self.update_memory(es, last_expansion, hs)
            cate_valid = get_cate_neighbors(cate_masks, edge_index, seed_mask,
                                            min_count=self.min_match)
            cate_valid = torch.stack(cate_valid, dim=0).bool()

            if i == n_iter - 1 and last_sample > 0:
                assert sample_group > 0
                expansion_i, expansion_score_i = [], []
                for j in range(sample_group):
                    expansion_ij, expansion_score_ij = \
                        self.output_layer(hs, es, cate_valid, last_sample,
                                          True)
                    expansion_i.append(expansion_ij)
                    expansion_score_i.append(expansion_score_ij)
            else:
                flag = is_all_sample if is_all_sample is not None else False
                expansion_i, expansion_score_i = \
                    self.output_layer(hs, es, cate_valid, self.n_expansion,
                                      flag)
                last_expansion = expansion_i
                seed_mask.scatter_(0, torch.cat(expansion_i, dim=0).view(-1), 1)
                for j in range(n_class):
                    cate_masks[j].scatter_(0, expansion_i[j], 1)

            outputs.append(expansion_score_i)
            expansions.append(expansion_i)
            hses.append(hs)
        return outputs, expansions, hses

    def update_memory(self, es, inps, hx=None):
        inp, mask = self.lookup_embedding(es, inps)
        inp = self.input_layer(hx, inp, mask)
        hx = self.memory_layer(hx, inp)
        return hx

    def lookup_embedding(self, es, inps):
        n_class, d_feature, device = len(inps), es.size(-1), es.device
        max_len = max([inp.size(0) for inp in inps])
        mask = torch.zeros([n_class, max_len], device=device)
        inputs = torch.zeros([n_class, max_len, d_feature], device=device)
        for i, inp in enumerate(inps):
            if inp.nelement() > 0:
                step = inp.size(0)
                inputs[i, :step] = es[inp]
                mask[i, :step] = 1
        return inputs, mask

    def inner_loss(self, hxes):
        criterion = nn.BCEWithLogitsLoss()
        hx_indice = torch.triu_indices(hxes[0].size(0), hxes[0].size(0), 1)
        losses = []
        for i, hx in enumerate(hxes):
            sim = self.sim_metric(hx, hx)
            sim = sim[hx_indice[0], hx_indice[1]]
            loss = criterion(sim, torch.zeros_like(sim))
            losses.append(loss)
        losses = torch.stack(losses, dim=0)
        return losses


class LNClassifier(nn.Module):
    def __init__(self, d_feature, n_class):
        super(LNClassifier, self).__init__()
        self.d_feature, self.n_class = d_feature, n_class
        self.fc = MLPLayer(d_feature, d_feature // 2, n_class)

    def forward(self, x):
        out = self.fc(x)
        return out
