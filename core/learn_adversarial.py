# coding=utf-8

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from .util import get_optimizer
from .evaluate import evaluate_generator
from .helper import prepare_discriminator_data, prepare_generator_data
from .discriminator import Discriminator


def train_generator(opt, gen, gen_opt, gen_scd, dis, prev_discriminators,
                    graph_data, seeds, n_epoch, n_iter, n_sample):
    gen.train()
    dis.eval()
    average_loss = 0
    for it in range(n_epoch):
        outs, expansions, _ = gen.sample(graph_data, seeds, n_iter, n_sample,
                                         sample_group=1, is_all_sample=False)
        with torch.no_grad():
            rewards = prepare_generator_data(dis, prev_discriminators,
                                             graph_data, expansions)
        gen_opt.zero_grad()
        pg_loss = gen.get_PGLoss(outs, expansions, rewards)
        pg_loss.backward()
        gen_opt.step()
        gen_scd.step()
        average_loss += pg_loss.item()

        if (it+1) % 5 == 0:
            print('Generator Loss: %.4f' % (pg_loss.item()))
    average_loss /= n_epoch
    print('Mean Generator Loss: %.4f' % (average_loss))


def train_discriminator(opt, dis, dis_opt, gen, graph_data, seeds,
                        n_steps, n_iter, epochs):
    dis.train()
    gen.eval()
    average_loss = 0
    for step in range(1, n_steps+1):
        with torch.no_grad():
            inps, targets = prepare_discriminator_data(gen, graph_data,
                                                       seeds, n_iter)
        dis_loss = 0
        for epoch in range(epochs):
            dis_opt.zero_grad()
            loss = dis.get_loss(graph_data, inps, targets)
            loss.backward()
            dis_loss += loss.item()
            dis_opt.step()
        average_loss += dis_loss
        if step % 5 == 0:
            print('Discriminator Loss: %.4f' % (dis_loss))
    average_loss /= n_steps
    print('Mean Discriminator Loss: %.4f' % (average_loss))


def adversarial_learn(opt, gen, graph_data, seeds):

    learn_epoch = 10
    print('\nStarting Adversarial Training...')
    gen_optimizer = get_optimizer(opt['optimizer'], gen.parameters(),
                                  opt['lr'], opt['decay'])
    gen_scd = lr_scheduler.StepLR(gen_optimizer, step_size=20*learn_epoch,
                                  gamma=0.9)

    prev_discriminators = []

    dis = Discriminator(opt)
    dis = dis.to(opt['device'])
    dis_optimizer = get_optimizer('rmsprop', dis.parameters(),
                                  opt['lr'], opt['decay'])
    print('\nWarm Training Discriminator :')
    # pre-training for 50 epochs at the each iteration beginning
    train_discriminator(opt, dis, dis_optimizer,
                        gen, graph_data, seeds, 50, 1, 1)
    for n_iter in range(1, opt['n_iter'] + 1):
        print('\n--------\nITER %d\n--------' % (n_iter))
        print('\nWarm Training Discriminator :')

        for n_epoch in range(1, learn_epoch + 1):
            print('\n--------\nEPOCH %d/%d\n--------' % (n_epoch, n_iter))

            # TRAIN DISCRIMINATOR
            print('\nAdversarial Training Discriminator :')
            train_discriminator(opt, dis, dis_optimizer,
                                gen, graph_data, seeds, 20, n_iter, 1)

            # TRAIN GENERATOR
            print('\nAdversarial Training Generator :')
            train_generator(opt, gen, gen_optimizer, gen_scd,
                            dis, prev_discriminators, graph_data, seeds,
                            20, n_iter, opt['n_expansion'] * 2)

            # EVALUATE GENERATOR
            evaluate_generator(gen, graph_data, seeds, n_iter=opt['n_iter'])

        prev_dis = Discriminator(opt)
        prev_dis = prev_dis.to(opt['device'])
        prev_dis.load_state_dict(dis.state_dict())
        prev_dis.eval()
        prev_discriminators.append(prev_dis)
