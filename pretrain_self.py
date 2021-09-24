# coding=UTF-8
import os

import torch
import torch.nn as nn
import numpy as np
import random
import argparse

from core.model import GBNEncoder, LNClassifier
from core.dataloader import load_graph
from core.sim_metric import EDSim, SDPSim, CosSim
from core.util import get_optimizer
from core.loss import edge_mask_loss, contrastive_loss


SIM_TABLE = {
    'ed': EDSim(),
    'sdp': SDPSim(),
    'cos': CosSim()
}


def regist_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--output_model_file', type=str, default='')
    parser.add_argument('--input_model_file', type=str, default='')
    parser.add_argument('--sim_metric', type=str, default='cos',
                        help='sim metric function,choose from list  ['
                        '"ed"(euclidean distance),"sdp"(scaled dot product)'
                        ',"cos"(cosine)]')
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--nl_weight', type=float, default=0.1)
    parser.add_argument('--local', action='store_true')

    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bias', action='store_true')

    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-3,
                        help='Weight decay for optimization')
    parser.add_argument('--feature_type', default='glove')
    parser.add_argument('--feature_dim', type=int, default=50)
    parser.add_argument('--edge_feature_dim', type=int, default=5)

    args = parser.parse_args()
    return args


def print_settings(opt):
    print('==========================Parameters==============================')
    print('Dataset Dir:\t\t', opt['data_dir'])
    print('Input Model Path:\t', opt['input_model_file'])
    print('Output Model Path:\t', opt['output_model_file'])
    print('Similarity Metric:\t', opt['sim_metric'])
    print('K Hop Neighbor:\t\t', opt['k_hop'])
    print('Is localized:\t\t', opt['local'])

    print('GNN Layer:\t\t', opt['n_layer'])
    print('Dropout:\t\t', opt['dropout'])
    print('Linear Bias:\t\t', opt['bias'])

    print('Random seed:\t\t', opt['seed'])

    print('Learning Epoch:\t\t', opt['n_epoch'])
    print('Optimizer:\t\t', opt['optimizer'])
    print('Learning Rate:\t\t', opt['lr'])
    print('Weight Decay:\t\t', opt['decay'])
    print('Feature Type:\t\t', opt['feature_type'])
    print('Feature Dim:\t\t', opt['feature_dim'])
    print('Edge Feature Dim:\t', opt['edge_feature_dim'])
    print('==================================================================')


def comprise_data(opt, encoder, weight):
    print('loading %s......' % opt['dataset'])
    pkl_path = 'graph_' + opt['feature_type'] + '.pkl'
    graph_data, graph = load_graph(opt, pkl_path, pyg=True)
    opt['n_class'] = len(graph.node_s.itol)

    graph_data = graph_data.to(opt['device'])
    graph_data.x = (graph_data.x[0].to(opt['device']),
                    graph_data.x[1].to(opt['device']))
    d_es = graph_data.x[0].size(-1)
    classifier = LNClassifier(d_es * 2, 1)
    classifier.to(opt['device'])
    parameters = [
        {'params': [p for p in encoder.parameters() if p.requires_grad]},
        {'params': [p for p in classifier.parameters() if p.requires_grad]}]
    optimizer = get_optimizer(opt['optimizer'], parameters,
                              opt['lr'], opt['decay'])
    n_epoch = opt['n_epoch'] * weight
    print('loaded!')
    return classifier, optimizer, graph_data, weight


def neighbor_learning_pretrain(opt, encoder, data):
    sim_metric = opt['sim_metric']
    es, ps = encoder(data)
    loss = opt['nl_weight'] * contrastive_loss((es, ps), data, sim_metric)
    return loss


def link_mask_pretrain(opt, encoder, classifier, data):
    x = data.x
    edge_attr = data.edge_attr
    edge_index = data.edge_index
    edge_size = edge_index.size(1)
    neg_size = int(edge_size * 0.1)

    indices = torch.randperm(edge_size, device=edge_index.device)
    masked_indices = indices[:neg_size]
    remain_indices = indices[neg_size:]

    output = encoder(x, edge_index[:, remain_indices],
                     edge_attr[remain_indices])
    loss = edge_mask_loss(output, data, masked_indices, classifier)
    return loss


def edge_mask(opt, encoder, batch, batch_id, ite):
    classifier, optimizer, data, weight = batch

    total_loss = 0
    for i in range(weight):
        encoder.train()
        optimizer.zero_grad()
        loss_nl = neighbor_learning_pretrain(opt, encoder, data)
        loss_lm = link_mask_pretrain(opt, encoder, classifier, data)
        loss = (loss_nl + loss_lm) / weight
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('Ite[%d]-Batch[%d]--loss:%.5f' % (ite, batch_id, total_loss, ))


if __name__ == '__main__':
    args = regist_parser()
    if args.seed:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # cuda device setting
    if args.cpu and args.device is not None or not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:%d' % args.device)
        if args.seed:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    opt = vars(args)
    if opt['local']:
        opt['k_hop'] = 1
    print_settings(opt)
    if opt['feature_type'] == 'bert':
        opt['feature_dim'] = 768
    device = opt['device']
    opt['sim_metric'] = SIM_TABLE[opt['sim_metric'].lower()].to(device)

    encoder = GBNEncoder(opt)
    if opt['input_model_file']:
        encoder.load_state_dict(torch.load(opt['input_model_file']+'.pth'))
    encoder = encoder.to(opt['device'])

    datasets = []
    with open(os.path.join(opt['data_dir'], 'unsupervised_dataset.txt'), 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            assert line and line[0]
            dataset = line[0]
            if len(line) > 1:
                weight = int(line[1])
            else:
                weight = 1
            datasets.append((dataset, weight))

    print('=================Do Pre-Train (Self-Supervised)===================')
    batches = []
    for dataset, weight in datasets:
        opt['dataset'] = os.path.join(opt['data_dir'], dataset)
        batches.append(comprise_data(opt, encoder, weight))

    for ite in range(1, opt['n_epoch'] + 1):
        for i, batch in enumerate(batches):
            edge_mask(opt, encoder, batch, i+1, ite)

    if opt['output_model_file']:
        print('write model file', opt['output_model_file'], '...')
        torch.save(encoder.cpu().state_dict(), opt['output_model_file']+'.pth')
