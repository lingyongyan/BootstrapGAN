# coding=utf-8
import torch


def prepare_discriminator_data(generator, graph_data, seeds, n_iter):
    _, expansions, _ = generator.sample(graph_data, seeds, n_iter,
                                        last_sample=n_iter * 10)

    # es, _ = generator.encoder(graph_data)
    previous_expansions = [[] for _ in range(len(seeds))]
    for step_expansions in expansions[:-1]:
        for i, expansion in enumerate(step_expansions):
            previous_expansions[i].append(expansion)
    if n_iter > 1:
        previous_expansions = [torch.cat(exp) for exp in previous_expansions]
    last_expansions = expansions[-1][-1]
    fake_inps = torch.cat(last_expansions)
    real_inps = []
    targets = []
    for cate, seed in enumerate(seeds):
        if n_iter > 1:
            prev = previous_expansions[cate]
            # perm = torch.randperm(prev.size(0))[:seed.size(0)]
            cate_inps = torch.cat([seed, prev])
        else:
            cate_inps = seed
        cate_targets = torch.full((cate_inps.size(0), ), cate,
                                  dtype=torch.long, device=cate_inps.device)
        real_inps.append(cate_inps)
        targets.append(cate_targets)
    real_inps = torch.cat(real_inps)
    targets = torch.cat(targets)
    perm = torch.randperm(targets.size(0))
    targets = targets[perm]
    real_inps = real_inps[perm]
    return (real_inps, fake_inps), targets


def prepare_generator_data(discriminator, prev_discriminators,
                           graph_data, expansions):
    inps = []
    for step_expansions in expansions[-1]:
        step_inps = []
        for cate_expansions in step_expansions:
            step_inps.append(cate_expansions)
        inps.append(step_inps)
    rewards = discriminator.batch_classify(graph_data, inps)

    prev_rewards = []
    for prev_dis, step_expansions in zip(prev_discriminators, expansions[:-1]):
        step_inps = []
        for cate_expansions in step_expansions:
            step_inps.append(cate_expansions)
        prev_rewards.extend(prev_dis.batch_classify(graph_data, [step_inps]))
    return prev_rewards, rewards


def prepare_discriminator_data_noboot(generator, graph_data, seeds, n_iter):
    _, expansions, _ = generator.sample(graph_data, seeds, n_iter,
                                        last_sample=10)

    previous_expansions = [[] for _ in range(len(seeds))]
    expansions[-1] = expansions[-1][-1]
    for step_expansions in expansions:
        for i, expansion in enumerate(step_expansions):
            previous_expansions[i].append(expansion)
    previous_expansions = [torch.cat(exp) for exp in previous_expansions]
    fake_inps = torch.cat(previous_expansions)
    real_inps = []
    targets = []
    for cate, seed in enumerate(seeds):
        cate_inps = seed
        cate_targets = torch.full((cate_inps.size(0), ), cate,
                                  dtype=torch.long, device=cate_inps.device)
        real_inps.append(cate_inps)
        targets.append(cate_targets)
    real_inps = torch.cat(real_inps)
    targets = torch.cat(targets)
    perm = torch.randperm(targets.size(0))
    targets = targets[perm]
    real_inps = real_inps[perm]
    return (real_inps, fake_inps), targets


def prepare_generator_data_single(discriminator, prev_discriminators,
                                  graph_data, expansions):
    inps = []
    for step_expansions in expansions[-1]:
        step_inps = []
        for cate_expansions in step_expansions:
            step_inps.append(cate_expansions)
        inps.append(step_inps)
    rewards = discriminator.batch_classify(graph_data, inps)

    prev_rewards = []
    for step_expansions in expansions[:-1]:
        step_inps = []
        for cate_expansions in step_expansions:
            step_inps.append(cate_expansions)
        prev_rewards.extend(discriminator.batch_classify(graph_data, [step_inps]))
    return prev_rewards, rewards
