# coding=UTF-8
"""
@Description: pre-process graph data to the pyG format
@Author: Lingyong Yan
@Date: 2019-08-06 22:42:04
@LastEditTime: 2019-08-28 11:57:42
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import torch
import torch.nn as nn


def sequence_mask(lens, max_len=None):
    """get a mask matrix from batch lens tensor

    :param lens:
    :param max_len:  (Default value = None)

    """
    if max_len is None:
        max_len = lens.max().item()
    batch_size = lens.size(0)

    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_broadcast = lens.unsqueeze(-1).expand_as(ranges)
    mask = ranges < lens_broadcast
    return mask


def mask_mean_weights(mask):
    new_mask = mask.float()
    sum_mask = new_mask.sum(dim=1, keepdim=True)
    indice = (sum_mask > 0).squeeze(1)
    new_mask[indice] /= sum_mask[indice]
    return new_mask

class CBOWEncoder(nn.Module):
    def __init__(self, vectors=None, vocab_size=None, emb_dim=None):
        super(CBOWEncoder, self).__init__()
        if vectors is not None:
            self.embed = nn.Embedding.from_pretrained(vectors)
        else:
            self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.requires_grad = False

    def forward(self, x, x_lens):
        embeddings = self.embed(x)
        masked_weights = mask_mean_weights(sequence_mask(x_lens))
        weighted_embedding = torch.bmm(masked_weights.unsqueeze(1), embeddings)
        return weighted_embedding.squeeze(1)
    