import torch
from torch.utils.data import DataLoader

from .roberta_dataset import RobertaDataset
from .tabert_dataset import TabertDataset
from .tapas_dataset import TapasDataset


def create_dataset(dataset, tokenizer, config):
    if dataset in {'tapas', 'bert'}:
        return TapasDataset(tokenizer, config['max_seq_length'], config['dataset_dir'], config['use_numtok'],
                            config['keep_origin'])
    if dataset == 'tabert':
        return TabertDataset(tokenizer, config['max_seq_length'])
    if dataset == 'roberta':
        return RobertaDataset(tokenizer, config['max_seq_length'], config['dataset_dir'], config['use_numtok'],
                              config['keep_origin'])


def create_sampler(datasets, shuffles, num_tasks, global_rank, seed):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset,
                                                      num_replicas=num_tasks, rank=global_rank, shuffle=shuffle,
                                                      seed=seed)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
