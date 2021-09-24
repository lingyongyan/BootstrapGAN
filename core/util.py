# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch.optim.lr_scheduler import LambdaLR


class HLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(HLoss, self).__init__()
        self.reduction = 1 if reduction == 'sum' else 0

    def forward(self, x, dim=-1):
        b = calc_entropy(x, dim)
        if dim != 0:
            reduction_dim = dim - 1
        else:
            reduction_dim = dim
        if self.reduction:
            b = b.sum(dim=reduction_dim)
        else:
            b = b.mean(dim=reduction_dim)
        return b.squeeze(-1)


def calc_entropy(logits, dim=-1):
    entropy = torch.sum(-1.0 * F.softmax(logits, dim=dim) * F.log_softmax(logits, dim=dim),
                        dim=dim, keepdim=True)
    return entropy


def scatter_mul(src, edge_index, edge_attr=None, dim=0):
    scatter_src = src.index_select(dim, edge_index[0])
    if edge_attr is not None:
        assert edge_index.size(1) == edge_attr.size(0)
        scatter_src = scatter_src * edge_attr.long()
    output = scatter_sum(scatter_src, edge_index[1], dim)
    return output


def cosine_sim(a, b, dim=-1, eps=1e-8):
    """calculate the cosine similarity and avoid the zero-division
    """
    a_norm = a / (a.norm(dim=dim)[:, None]).clamp(min=eps)
    b_norm = b / (b.norm(dim=dim)[:, None]).clamp(min=eps)
    if len(a.shape) <= 2:
        sim = torch.mm(a_norm, b_norm.transpose(1, 0))
    else:
        sim = torch.einsum('ijk, lmk->iljm', (a_norm, b_norm))
    return sim


def get_optimizer(name, parameters, lr, weight_decay=0.):
    """initialize parameter optimizer
    """
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay,
                               momentum=0.9)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr,
                                   weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr,
                                   weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, last_epoch=-1,
                                    min_ratio=0.0):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(min_ratio, float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def predict_confidence(predicts):
    entropy = -torch.mean(predicts * predicts.log(), dim=-1)
    max_entropy = entropy.max()
    confidence = 1 - entropy / max_entropy
    return confidence
