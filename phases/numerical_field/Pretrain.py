import setproctitle

setproctitle.setproctitle("Pretrain")
import argparse
import os

os.sys.path.insert(0, '')
try:
    import ruamel.yaml as yaml
except:
    import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from phases.numerical_field.models.roberta_num import RobertaNum
from phases.numerical_field.models.tapas_num import TapasNum
from phases.numerical_field.models.bert_num import BertNum
from transformers import RobertaTokenizer, TapasTokenizer

import phases.numerical_field.utils as utils
from phases.numerical_field.dataset import create_dataset, create_sampler, create_loader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from utils import setup_seed,WandbWrapper


def get_coef(config, alpha):
    if config['use_distrib']:
        if config['use_mlm']:
            if config['use_distill']:
                return (1 - alpha, alpha, 1)
            return (1, 0, 1)
        if config['use_distill']:
            return (0, 1, 1)
        return (0, 0, 1)
    if config['use_mlm']:
        if config['use_distill']:
            return (1 - alpha, alpha, 0)
        return (1, 0, 0)
    if config['use_distill']:
        return (0, 1, 0)
    return (0, 0, 0)


global_steps = 0


def train(model, data_loader, optimizer, epoch, device, scheduler, config, wandb):
    # train
    model.train()
    global global_steps

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_distill', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_distrib', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = config['step_size']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    warm_up_steps = config['warm_up_steps']
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (inputs, number_lists, table_distributions) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        mcontext = model.no_sync if i % gradient_accumulation_steps != 0 else utils.nullcontext
        with mcontext():
            for k, v in inputs.items():
                inputs[k] = v.to(device, non_blocking=True)
            table_distributions = table_distributions.to(device, non_blocking=True)
            length = torch.where(inputs['attention_mask'] != 0)[1].max() + 1
            for k, v in inputs.items():
                inputs[k] = v[:, :length]
            decay = min(1, 0.9 * global_steps / warm_up_steps + 0.1)
            alpha = config['alpha'] * decay
            loss_distill, loss_mlm, loss_distrib = model(number_list=number_lists,
                                                         table_distributions=table_distributions, **inputs)
            coef = get_coef(config, alpha)
            loss = coef[0] * loss_mlm + coef[1] * loss_distill + 0.2 * coef[2] * loss_distrib
            metric_logger.update(loss_distill=loss_distill.item())
            metric_logger.update(loss_mlm=loss_mlm.item())
            metric_logger.update(loss_distrib=loss_distrib.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=decay * optimizer.param_groups[0]["lr"])
            agg_loss = decay * loss / gradient_accumulation_steps
            agg_loss.backward()
        # 轮数为accumulation_steps整数倍的时候，传播梯度，并更新参数
        if i % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({'loss_distill': loss_distill.item(), 'loss_mlm': loss_mlm.item(), 'loss': loss.item(),
                       'lr': decay * optimizer.param_groups[0]["lr"]},
                      step=global_steps)
            if global_steps % step_size == 0:
                scheduler.step()
            if 'save_by_steps' in config and global_steps % config['save_by_steps'] == config['save_by_steps'] - 1:
                metric_logger.synchronize_between_processes()
                if utils.is_main_process():
                    print("Averaged stats:", metric_logger.global_avg())
                    print('============================================')
                    log_stats = {
                        **{f'train_{k}': "{:.5f}".format(meter.global_avg) for k, meter in
                           metric_logger.meters.items()},
                        'epoch': '%.2f' % (global_steps / (len(data_loader) / gradient_accumulation_steps)),
                    }
                    save_obj = {
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%s.pth' % log_stats['epoch']))
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                metric_logger.reset()
            global_steps += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def collate_fn(data):
    inputs = {k: torch.stack([x[0][k] for x in data]) for k in data[0][0].keys()}
    number_lists = []
    for x in data:
        number_lists.extend(x[1])
    table_distributions = torch.stack([x[2] for x in data])
    return inputs, number_lists, table_distributions


def main(args, config):
    utils.init_distributed_mode(args)
    wandb = WandbWrapper(utils.is_main_process())
    # import wandb
    wandb.init(name=config['run_name'],config=config, notes=os.environ.get('WANDB_RUN_NOTES', None))
    wandb.config.update({'aml_user': os.environ.get("USER", None),
                         'exp_name': os.environ.get("EXP_NAME", None),
                         'commit_hash': os.environ.get("COMMIT_HASH", None),
                         'cluster': os.environ.get("CLUSTER_NAME", None),
                         'git_branch': os.environ.get("GIT_BRANCH", None)
                         })
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    setup_seed(seed)
    cudnn.enabled = False
    # cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['epochs']
    if args.encoder == 'roberta':
        if os.path.exists('/storage/tuna-models/roberta.large'):
            roberta_path='/storage/tuna-models/roberta.large'
        elif os.path.exists('data/ckpt/roberta.large'):
            roberta_path='data/ckpt/roberta.large'
        else:
            raise NotImplementedError
        tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
    elif args.encoder in {'tapas', 'bert'}:
        tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-masklm')
    tokenizer.add_tokens(["[NUM]"])
    tokenizer.num_token_id = len(tokenizer) - 1
    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset(args.encoder, tokenizer, config)]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank, args.seed)
    else:
        samplers = [None]

    train_loader = create_loader(datasets, samplers,
                                 batch_size=[config['batch_size']],
                                 num_workers=[4], is_trains=[True],
                                 collate_fns=[collate_fn])[0]
    if 'save_by_steps' in config:
        assert config['save_by_steps'] < len(train_loader) / config['gradient_accumulation_steps']
    #### Model #### 
    print("Creating model")
    if args.encoder == 'roberta':
        model = RobertaNum(config=config, tokenizer=tokenizer)
    elif args.encoder == 'tapas':
        model = TapasNum(config=config, tokenizer=tokenizer)
    elif args.encoder == 'bert':
        model = BertNum(config=config, tokenizer=tokenizer)
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = AdamW(model.parameters(), **arg_opt)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, **arg_sche)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

        print(model.load_state_dict(state_dict, strict=False))
        print('load checkpoint from %s' % args.checkpoint)

    model_layers = getattr(model.encoder, args.encoder).encoder.layer
    if args.tune_layers != -1:
        assert args.tune_layers >= 0 and args.tune_layers <= len(model_layers)
        for i in range(args.tune_layers, len(model_layers)):
            for param in model_layers[i].parameters():
                param.requires_grad = False

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model = model.to(device)

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        train_stats = train(model, train_loader, optimizer, epoch, device, lr_scheduler, config, wandb)
        if ('save_by_steps' not in config and epoch % config['save_by_epochs'] == config[
            'save_by_epochs'] - 1 or epoch == max_epoch - 1) and utils.is_main_process():
            print('============================================')
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument("--config", type=str, default='phases/numerical_field/configs/small/Pretrain_roberta.yaml')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument("--run_name", type=str, default='pretrain', help='wandb runname')
    parser.add_argument('--use_numtok', default=1, type=int)
    parser.add_argument('--keep_origin', default=1, type=int)
    parser.add_argument('--use_prompt', default=0, type=int)
    parser.add_argument('--use_rank', default=1, type=int)
    parser.add_argument('--use_mlm', default=1, type=int)
    parser.add_argument('--use_distill', default=1, type=int)
    parser.add_argument('--use_distrib', default=0, type=int)
    parser.add_argument('--use_huber', default=0, type=int)
    parser.add_argument('--use_logflat', default=0, type=int)
    parser.add_argument('--use_firsttoken', default=0, type=int)
    parser.add_argument('--use_regression', default=1, type=int)
    parser.add_argument('--tune_layers', default=-1, type=int, help="number of layers to be tuned. \
                                                                    default -1, means fix nothing; \
                                                                    0, means fix all transformer layers; \
                                                                    6, means fix the last len(layers)-6 layers.")
    parser.add_argument('--numbed_ckpt', default='', type=str)
    parser.add_argument('--numbed_model', default='', type=str, help="'zero' to use zero numbed")
    parser.add_argument("--kept_keys", type=str, default='', help="param of Numtok.tokenize")
    parser.add_argument('--output_dir', default='data/ckpt/Roberta/', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    args.encoder = config['encoder']
    assert args.use_mlm or args.use_distill or args.use_distrib
    print('use_mlm:%s;use_distill:%s;use_distrib:%s' % (args.use_mlm, args.use_distill, args.use_distrib))
    for k in vars(args):
        if k not in {'numbed_ckpt', 'numbed_model'}:
            config[k] = getattr(args, k)
    if args.numbed_ckpt != '':
        if args.numbed_ckpt == 'random':
            config['numbed_ckpt'] = ''
        else:
            config['numbed_ckpt'] = args.numbed_ckpt
    if args.numbed_model != '':
        config['numbed_model'] = args.numbed_model
    if 'save_by_epochs' not in config: config['save_by_epochs'] = 1
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
