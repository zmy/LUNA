import setproctitle

setproctitle.setproctitle("TAT")
import argparse
import os, json

os.sys.path.insert(0, '')
from pathlib import Path
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from number_tokenizer import prepare_tokenizer
from utils import to_device, setup_seed, WandbWrapper
from phases.downstream_tasks.TAT.options import add_train_args, add_bert_args, DEFAULT_DATA_DIR, TABLE_ENCODERS, \
    DEFAULT_MODEL_DIR
from phases.downstream_tasks.TAT.tagop.roberta_num import RobertaNum
from phases.downstream_tasks.TAT.data.roberta_dataset import collate as roberta_collate
from phases.downstream_tasks.TAT.data.tapas_dataset import collate as tapas_collate
from phases.downstream_tasks.TAT.tagop.optimizer import BertAdam as Adam
from phases.downstream_tasks.TAT.tagop.tapas_num import TapasNum
from phases.downstream_tasks.TAT.tagop.bert_num import BertNum
import logging
import pickle as pk
from contextlib import nullcontext


def prepare_data(data, is_train, args, collate_fn):
    data_sample = DistributedSampler(data, num_replicas=args.world_size, rank=args.rank, seed=args.seed)
    batch_size = args.batch_size_per_node if is_train else args.eval_batch_size_per_node
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=1, drop_last=is_train,
                             sampler=data_sample, collate_fn=collate_fn)
    return data_loader


def load_data(args, tokenizer):
    name = ['ori', 'num', 'both', 'both']
    with open(os.path.join(args.input_dir,
                           f"{'roberta' if args.encoder == 'roberta' else 'tapas'}_{name[args.use_numtok]}_cache_train.pkl"),
              'rb') as f:
        train_data = np.array(pk.load(f), dtype=np.object)
    with open(os.path.join(args.input_dir,
                           f"{'roberta' if args.encoder == 'roberta' else 'tapas'}_{name[args.use_numtok]}_cache_dev.pkl"),
              'rb') as f:
        dev_data = np.array(pk.load(f), dtype=np.object)
    kept_keys = tuple(args.kept_keys.split(',')) if args.kept_keys != '' else ()
    if args.encoder == 'roberta':
        collate = lambda x: roberta_collate(x, tokenizer, kept_keys)
    elif args.encoder in {'tapas', 'bert'}:
        collate = lambda x: tapas_collate(x, tokenizer, kept_keys, args.encoder)

    train_dataloader = prepare_data(train_data, True, args, collate)
    dev_dataloader = prepare_data(dev_data, False, args, collate)

    return train_dataloader, dev_dataloader


def choose_model(args):
    tokenizer = prepare_tokenizer('roberta' if args.encoder == 'roberta' else 'tapas', model_dir=args.model_dir,
                                  redirect_huggingface_cache=args.redirect_huggingface_cache)
    if args.encoder == 'roberta':
        clazz = RobertaNum
        hidden_size = 1024
    elif args.encoder == 'tapas':
        clazz = TapasNum
        hidden_size = 768
    elif args.encoder == 'bert':
        clazz = BertNum
        hidden_size = 768
    network = clazz(
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        dropout_prob=args.dropout,
        use_newtag=args.use_newtag,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        model_dir=args.model_dir,
        redirect_huggingface_cache=args.redirect_huggingface_cache,
        use_prompt=args.use_numtok==3,
    )
    return tokenizer, network


global_train_step = 0


def train(args, logger, ddp_network, train_dataloader, epoch, device, optim, writer):
    global global_train_step
    ddp_network.train()
    train_dataloader.sampler.set_epoch(epoch)
    loss_tracer = defaultdict(float)
    keys = ['operator_prediction_loss', 'scale_prediction_loss', \
            'tag_prediction_loss', 'top_2_order_prediction_loss', 'loss', 'cnt']
    step_cnt = int(len(train_dataloader.dataset) / (args.world_size * args.batch_size_per_node))
    for step, data in enumerate(train_dataloader):
        del data['question_ids']
        data = to_device(data, device)
        mcontext = ddp_network.no_sync if global_train_step % args.gradient_accumulation_steps != 0 else nullcontext
        with mcontext():
            operator_prediction_loss, scale_prediction_loss, tag_prediction_loss, top_2_order_prediction_loss = \
                ddp_network(**data)['loss']
            loss = operator_prediction_loss + scale_prediction_loss + tag_prediction_loss + top_2_order_prediction_loss
            agg_loss = loss / args.gradient_accumulation_steps
            agg_loss.backward()
        if global_train_step % args.gradient_accumulation_steps == 0:
            optim.step()
            optim.zero_grad()

        for _name, _loss in zip(keys, \
                                [operator_prediction_loss.item(), scale_prediction_loss.item(), \
                                 tag_prediction_loss.item(), top_2_order_prediction_loss.item(), loss.item(), 1]):
            loss_tracer[_name] += _loss
        if writer.is_main_process and step % args.log_per_updates == args.log_per_updates - 1:
            msg = "Epoch:%d, step:%d/%d" % (epoch, step, step_cnt)
            log = {}
            for _name in keys[:-1]:
                msg += ', %s:%.6f' % (_name, loss_tracer[_name] / loss_tracer['cnt'])
                log['%s' % _name] = loss_tracer[_name] / loss_tracer['cnt']
            log = {'train': log}
            writer.log(log, step=global_train_step)
            logger.info(msg)
            loss_tracer = defaultdict(float)

        global_train_step += 1


def dev(args, logger, ddp_network, dev_dataloader, epoch, device, writer):
    ddp_network.eval()
    metrics = ddp_network.module._metrics
    metrics.reset()
    loss_tracer = defaultdict(float)
    keys = ['operator_prediction_loss', 'scale_prediction_loss', \
            'tag_prediction_loss', 'top_2_order_prediction_loss', 'loss', 'cnt']
    with torch.no_grad():
        for step, data in enumerate(dev_dataloader):
            if not args.render_badcase:
                del data['question_ids']
            data = to_device(data, device)
            output = ddp_network(**data, dev=True)
            operator_prediction_loss, scale_prediction_loss, tag_prediction_loss, top_2_order_prediction_loss = output[
                'loss']
            loss = operator_prediction_loss + scale_prediction_loss + tag_prediction_loss + top_2_order_prediction_loss

            for _name, _loss in zip(keys, \
                                    [operator_prediction_loss.item(), scale_prediction_loss.item(), \
                                     tag_prediction_loss.item(), top_2_order_prediction_loss.item(), loss.item(), 1]):
                loss_tracer[_name] += _loss
            if args.rank == 0 and step % args.log_per_updates == args.log_per_updates - 1:
                msg = "Epoch:%d, step:%d, example_gt:%s, example_pred:%s" % (
                    epoch, step, str(data['gold_answers'][0]), str(output['answer'][0]))
                logger.info(msg)

        t = torch.tensor([loss_tracer[key] for key in keys],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        for key, value in zip(keys, t):
            loss_tracer[key] = value

        if args.render_badcase:
            os.makedirs(os.path.join(args.save_dir, 'badcases', 'e%d' % epoch), exist_ok=True)
            with open(os.path.join(args.save_dir, 'badcases', 'e%d' % epoch, 'case%d.json' % args.rank), 'w',
                      encoding='utf8') as f:
                json.dump(metrics.badcase, f)
            dist.barrier()
            assert args.world_size == torch.cuda.device_count()
            if args.rank == 0:
                badcase = {'aop_cnt': defaultdict(int), 'cases': {}}
                for i in range(args.world_size):
                    with open(os.path.join(args.save_dir, 'badcases', 'e%d' % epoch, 'case%d.json' % i), 'r',
                              encoding='utf8') as f:
                        _badcase = json.load(f)
                        for k, v in _badcase['aop_cnt'].items():
                            badcase['aop_cnt'][k] += v
                        badcase['cases'].update(_badcase['cases'])
                    os.remove(os.path.join(args.save_dir, 'badcases', 'e%d' % epoch, 'case%d.json' % i))
                badcase['berror_rate'] = {}
                for k, v in badcase['aop_cnt'].items():
                    badcase['berror_rate'][k] = sum([1 for x in badcase['cases'].values() if x[0][1] == k]) / v
                with open(os.path.join(args.save_dir, 'badcases', 'e%d' % epoch, 'case.json'), 'w',
                          encoding='utf8') as f:
                    json.dump(badcase, f, sort_keys=True, indent=4)
        metrics.synchronize_between_processes()
        if args.rank == 0:
            msg = "Epoch:%d" % epoch
            log = {}
            for _name in keys[:-1]:
                msg += ', %s:%.6f' % (_name, loss_tracer[_name] / loss_tracer['cnt'])
                log['%s' % _name] = loss_tracer[_name] / loss_tracer['cnt']
            msg += ', ' + str(metrics)
            log['em'] = metrics._total_em / (metrics._count + 1e-7)
            log['f1'] = metrics._total_f1 / (metrics._count + 1e-7)
            log['math_em'] = metrics._math_em / (metrics._math_cnt + 1e-7)
            log = {'dev': log, 'epoch': epoch}
            logger.info(msg)
            if writer.is_main_process:
                writer.log(log, step=global_train_step)
            return msg


def main(args):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    logger.info("environments setting!")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=dist.Backend.NCCL, init_method="env://", world_size=args.world_size, rank=args.rank)
    dist.barrier()
    writer = WandbWrapper(args.rank == 0 and not args.dev_only)
    # import wandb
    writer.init(name=args.run_name,config=args, notes=os.environ.get('WANDB_RUN_NOTES', None))
    writer.config.update({'aml_user': os.environ.get("USER", None),
                          'exp_name': os.environ.get("EXP_NAME", None),
                          'commit_hash': os.environ.get("COMMIT_HASH", None),
                          'cluster': os.environ.get("CLUSTER_NAME", None),
                          'git_branch': os.environ.get("GIT_BRANCH", None)
                          })
    device = 'cuda'
    setup_seed(args.seed + args.rank)

    # if args.rank == 0 and not args.dev_only:
    #     writer = SummaryWriter(args.tblog_dir)
    # else:
    #     writer = None

    logger.info(f"Build {args.encoder} model. numtok = {args.use_numtok}")
    tokenizer, network = choose_model(args)

    logger.info(f"loading data")
    train_dataloader, dev_dataloader = load_data(args, tokenizer)

    logger.info(f"Build optimizer")
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in network.encoder.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': args.bert_weight_decay, 'lr': args.bert_learning_rate},
        {'params': [p for n, p in network.encoder.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.bert_learning_rate},
        {'params': [p for n, p in network.named_parameters() if not n.startswith("encoder.")],
         "weight_decay": args.weight_decay, "lr": args.learning_rate}
    ]
    optim = Adam(optimizer_parameters,
                 lr=args.learning_rate,
                 warmup=args.warmup,
                 t_total=int(args.max_epoch * len(train_dataloader.dataset) / (
                         args.world_size * args.batch_size_per_node * args.gradient_accumulation_steps)),
                 max_grad_norm=args.grad_clipping,
                 schedule=args.warmup_schedule)

    logger.info('init...')
    if args.weights_path != '':
        assert os.path.exists(args.weights_path)
        weights_dict = torch.load(args.weights_path, map_location='cpu')['model']
        for k, v in list(weights_dict.items()):
            prefix = k.split('.', 1)[0]
            if prefix == 'number_model':
                new_k = 'numbed.' + k.split('.', 1)[1]
                weights_dict[new_k] = v
                del weights_dict[k]
            elif prefix == 'encoder':
                second_prefix = k.split('.')[1]
                if second_prefix == args.encoder:
                    new_k = '.'.join([prefix] + k.split('.')[2:])
                    weights_dict[new_k] = v
                del weights_dict[k]
            else:
                del weights_dict[k]
        print(network.load_state_dict(weights_dict, strict=False))

    if args.ckpt != '':
        assert os.path.exists(args.ckpt)
        network.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=True)

    network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network).to(device)
    ddp_network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[args.local_rank],
                                                            find_unused_parameters=True)
    if not args.dev_only:
        logger.info("start training")
        for epoch in range(args.max_epoch):
            train(args, logger, ddp_network, train_dataloader, epoch, device, optim, writer)
            msg = dev(args, logger, ddp_network, dev_dataloader, epoch, device, writer)
            if args.rank == 0:
                assert msg is not None
                with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                    f.write(msg + "\n")
                torch.save(network.state_dict(), os.path.join(args.save_dir, 'checkpoint_%02d.pth' % epoch))
    else:
        logger.info("start developping")
        dev(args, logger, ddp_network, dev_dataloader, -1, device, None)


def load_args():
    parser_op = argparse.ArgumentParser("TagOp training.")
    add_train_args(parser_op)
    add_bert_args(parser_op)
    parser_op.add_argument("--encoder", type=str, default='roberta', choices=TABLE_ENCODERS,
                           help='which encoder to use, default roberta')
    parser_op.add_argument("--input_dir", type=str, default=DEFAULT_DATA_DIR, help='where the data is')
    parser_op.add_argument("--run_name", type=str, default='tatqa', help='wandb runname')
    parser_op.add_argument("--weights_path", type=str, default='',
                           help='if empty(default), huggingface pretrained ckpt will be used; else, phase 1 will be used')
    parser_op.add_argument("--ckpt", type=str, default='',
                           help='if empty(default), nothing will happen; else, a TATQA ckpt will be used')
    parser_op.add_argument("--save_dir", default=os.path.join(DEFAULT_MODEL_DIR, 'checkpoint', "tatqa_roberta_num"),
                           type=str, help='where to save results')
    parser_op.add_argument("--tblog_dir", default="",
                           type=str, help='where to save eventfile, default same with save_dir')
    parser_op.add_argument("--use_numtok", type=int, default=1,
                           help='whether to use numtok, default 1; 0 means origin; 2 means both origin and numtok; 3 means prompt')
    parser_op.add_argument("--use_newtag", type=int, default=0,
                           help='whether to use mean pooling for tagging probability, default 0')
    parser_op.add_argument("--dev_only", type=int, default=0, help='whether to evaluation only, default 0')
    parser_op.add_argument("--render_badcase", type=int, default=0, help='whether to render badcases, default 0')
    parser_op.add_argument("--kept_keys", type=str, default='batch_token_ids,batch_seq_len',
                           help="param of Numtok.collate")
    parser_op.add_argument("--model_name", type=str, default='',
                           help="if empty(default),default model will be used;'zero' to use zero numbed")
    parser_op.add_argument("--checkpoint_path", type=str, default='',
                           help="if empty(defaut), default numbed will be used; if 'random', random numbed will be used")
    args = parser_op.parse_args()

    if args.model_name == '':
        args.model_name = 'CharLSTM_base' if args.encoder in {'tapas', 'bert'} else 'CharLSTM_large'
    if args.model_name != 'zero':
        if args.checkpoint_path != '':
            if args.checkpoint_path == 'random':
                args.checkpoint_path = ''
        else:
            assert args.model_name in {'CharLSTM_base', 'CharLSTM_large'}
            args.checkpoint_path = '/storage/tuna-models/numbed-ckpt/%s.pt' % args.model_name

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.tblog_dir == "":
        args.tblog_dir = args.save_dir
    elif args.tblog_dir == "auto":
        args.tblog_dir = os.path.join('/tb_logs', args.save_dir.strip('/').split('/')[-1])
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = load_args()
    main(args)
