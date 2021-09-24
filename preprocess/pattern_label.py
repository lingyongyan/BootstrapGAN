# coding=utf-8
import os
import argparse


def regist_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    return args


def get_patterns(data_dir):
    patterns = []
    with open(os.path.join(data_dir, 'patterns.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                patterns.append(line)
    return patterns


def label_patterns(data_dir):
    patterns = get_patterns(data_dir)
    labels = []
    for pattern in patterns:
        assert pattern.count('_') == 1
        if pattern[0] == '_':
            labels.append(1)
        elif pattern[-1] == '_':
            labels.append(3)
        else:
            labels.append(2)
    with open(os.path.join(data_dir, 'pattern_labels.txt'), 'w') as f:
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            f.write(str(i) + '\t' + str(label) + '\n')


if __name__ == '__main__':
    args = regist_parser()
    opts = vars(args)

    label_patterns(opts['dir'])
