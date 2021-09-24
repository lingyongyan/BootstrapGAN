# coding=utf-8
import os
import copy

opt = dict()
opt['dataset'] = 'dataset/OntoNotes'

devices = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_seeds(seed_file):
    seeds = []
    with open(seed_file, 'r') as f:
        for line in f:
            seeds.append(int(line.strip()))
    return seeds


def generate_command(opt, log_path):
    cmd = 'python -u bootstrap.py --local'
    for opt, val in opt.items():
        if val is not None and val != '':
            cmd += ' --' + opt + ' ' + str(val)
    cmd = 'nohup ' + cmd + ' > ' + log_path + ' 2>&1 &'
    return cmd


def run(opt, log_path):
    opt_ = copy.deepcopy(opt)
    cmd = generate_command(opt_, log_path)
    print('\n------Run following command------\n%s' % cmd)
    os.system(cmd)


for i in range(9):
    device = devices[i%9]
    opt['device'] = device
    log_path = 'logs/onto_abl_imb1_' + str(i+1) + '.txt'
    run(opt, log_path)
