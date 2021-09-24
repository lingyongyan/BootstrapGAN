# coding=utf-8

import torch
import torch.nn as nn

from .evaluate import eval_encoder_classifier
from .util import get_optimizer, get_linear_schedule_with_warmup
from .loss import contrastive_loss
from .learn_multi_view import multi_view_learn, update_s
from .learn_adversarial import adversarial_learn


def pretrain(opt, gen, graph_data, seeds):
    classifier = pretrain_encoder(opt, gen.encoder, graph_data, seeds)
    update_s(opt, gen, classifier, graph_data, seeds)
    return classifier


def pretrain_encoder(opt, encoder, graph_data, seeds):
    n_class = opt['n_class']
    d_layer = opt['layer_dim']
    classifier = nn.Linear(d_layer, n_class)
    classifier.to(opt['device'])
    criterion = torch.nn.CrossEntropyLoss()
    sim_metric = opt['sim_metric']
    parameters = [
        {'params': [p for p in encoder.parameters() if p.requires_grad],
         'lr': opt['lr'] * 5,
         'weight_decay': opt['decay']},
        {'params': [p for p in classifier.parameters() if p.requires_grad]}]
    optimizer = get_optimizer(opt['optimizer'], parameters,
                              opt['lr'] * 5, opt['decay'])
    steps = opt['init_encoder_epoch']
    # warm_steps = steps * 0.1
    # scheduler = get_linear_schedule_with_warmup(optimizer, warm_steps, steps)
    eval_encoder_classifier(encoder, classifier, graph_data, seeds)

    seed = torch.cat(seeds, dim=0)
    seed_label = graph_data.y[seed]
    for i in range(1, steps+1):
        encoder.train()
        classifier.train()
        optimizer.zero_grad()
        es, ps = encoder(graph_data)
        logits = classifier(es[seed])
        sp_loss = criterion(logits, seed_label)
        un_loss = contrastive_loss((es, ps), graph_data, sim_metric)
        loss = sp_loss + 0.1 * un_loss
        loss.backward()
        optimizer.step()
        print('Pre-train--Step: %d, loss:%.4f' %
              (i, loss.item()))
        if i % 50 == 0:
            eval_encoder_classifier(encoder, classifier, graph_data, seeds)
    return classifier


def fine_tune(opt, gen, graph_data, seeds, dev_seeds=None):
    classifier = pretrain(opt, gen, graph_data, seeds)
    if opt['method'] == 'adv':
        adversarial_learn(opt, gen, graph_data, seeds)
    else:
        multi_view_learn(opt, gen, classifier, graph_data, seeds)
