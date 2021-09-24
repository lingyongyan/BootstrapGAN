# coding=UTF-8
import os

import numpy as np
import random
import argparse
import torch

from core.sim_metric import EDSim, SDPSim, CosSim
from core.learn_entry import fine_tune, pretrain_encoder
from core.dataloader import load_graph, load_seed
from core.generator import Generator
from core.discriminator import Discriminator
from core.evaluate import evaluate_generator
from core.learn_multi_view import update_s


SIM_TABLE = {
    'ed': EDSim(),
    'sdp': SDPSim(),
    'cos': CosSim()
}


def regist_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/CoNLL')
    parser.add_argument('--input_model_file', type=str, default='')
    parser.add_argument('--output_model_file', type=str, default='')
    parser.add_argument('--method', type=str, default='multi_view')
    parser.add_argument('--sim_metric', type=str, default='cos',
                        help='sim metric function,choose from list  ['
                        '"ed"(euclidean distance),"sdp"(scaled dot product)'
                        ',"cos"(cosine)]')
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--min_match', type=int, default=1)
    parser.add_argument('--n_expansion', type=int, default=10)
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--mean_updated', action='store_true')

    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bias', action='store_true')

    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.0)

    parser.add_argument('--init_encoder_epoch', type=int, default=100)
    parser.add_argument('--init_decoder_epoch', type=int, default=100)
    parser.add_argument('--encoder_epoch', type=int, default=50)
    parser.add_argument('--decoder_epoch', type=int, default=50)
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-3,
                        help='Weight decay for optimization')
    parser.add_argument('--feature_type', default='glove')
    parser.add_argument('--feature_dim', type=int, default=50)
    parser.add_argument('--edge_feature_dim', type=int, default=5)

    args = parser.parse_args()
    return args


def load_tables(file_path):
    table = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            if len(line) > 1:
                table.append(line[0] + '(' + line[1] + ')')
            else:
                table.append(line[0])
    return table


def print_case(name, model, graph, graph_data, seeds,
               entity_table, label_table):
    _, expansions, _ = model(graph_data, seeds, 20)
    n = len(label_table)

    errors = [[0 for _ in range(n)] for _ in range(n)]
    with open(name+'.tsv', 'w') as f:
        for it, _expansions in enumerate(expansions):
            f.write('=======Iteration %d=======\t \t \n' % it)
            for i, expansion in enumerate(_expansions):
                original_i = int(graph.node_s.itol[i])
                original_label = label_table[original_i]
                r_set = set()
                f_set = set()
                for en in expansion.cpu().tolist():
                    original_en = entity_table[int(graph.node_s.itos[en])]
                    if graph_data.m_y[en][i] == 1:
                        r_set.add(original_en)
                    else:
                        f_set.add(original_en)
                        true_i = int(graph.node_s.itol[graph_data.y[en].cpu().item()])
                        errors[true_i][original_i] += 1
                f.write(str(original_label) + '\t' + ','.join(r_set) + '\t' + ','.join(f_set) + '\n')
            f.write('\n')

        f.write('\n==========errors==========\n')
        f.write(' OOOOO ')
        for i in range(n):
            f.write('\t'+label_table[i])
        f.write('\n')
        total = 0
        for i in range(n):
            f.write(label_table[i] + '\t' + '\t'.join(list(map(str, errors[i]))))
            total += sum(errors[i])
            f.write('\n')
        f.write('TOTAL ERRORS: %d'%total)


if __name__ == '__main__':
    args = regist_parser()
    if args.seed:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # cuda device setting
    if args.cpu and args.device is not None or not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:%d' % args.device)

    opt = vars(args)
    if opt['local']:
        opt['k_hop'] = 1
    if opt['feature_type'] == 'bert':
        opt['feature_dim'] = 768
    device = opt['device']
    opt['sim_metric'] = SIM_TABLE[opt['sim_metric'].lower()].to(device)

    pkl_path = 'graph_' + opt['feature_type'] + '.pkl'
    graph_data, graph = load_graph(opt, pkl_path, pyg=True)
    seed_file = opt['dataset'] + '/seeds.txt'
    seeds = load_seed(graph.node_s, seed_file)
    seeds = [torch.LongTensor(seed) for seed in seeds]
    opt['n_class'] = len(graph.node_s.itol)

    generator = Generator(opt)

    graph_data = graph_data.to(opt['device'])
    graph_data.x = (graph_data.x[0].to(opt['device']),
                    graph_data.x[1].to(opt['device']))
    seeds = [seed.to(opt['device']) for seed in seeds]
    generator = generator.to(opt['device'])

    entity_table = load_tables(opt['dataset'] + '/entities.txt')
    label_table = load_tables(opt['dataset'] + '/labels.txt')

    classifier = pretrain_encoder(opt, generator.encoder, graph_data, seeds)
    update_s(opt, generator, classifier, graph_data, seeds)
    evaluate_generator(generator, graph_data, seeds, n_iter=20)
    dataset_name = os.path.split(opt['dataset'])[1].lower()
    print_case('results/original_' + dataset_name, generator, graph,
               graph_data, seeds, entity_table, label_table)

    print('++++++adversarial learning++++++++')
    if opt['input_model_file']:
        pkl_path = opt['input_model_file']
        generator = generator.cpu()
        generator.load_state_dict(torch.load(pkl_path))
    generator = generator.to(opt['device'])
    evaluate_generator(generator, graph_data, seeds, n_iter=20)
    print_case('results/' + dataset_name, generator, graph, graph_data, seeds,
               entity_table, label_table)
