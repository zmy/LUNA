from fileinput import filename
import setproctitle

setproctitle.setproctitle("TabFact")
import argparse
import logging
import os,json

from torch.optim.adamw import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from datasets import load_metric
from .data_util import prepare_data, read_text_as_pandas_table, roberta_string_tokenize, roberta_table_tokenize
from number_tokenizer.numtok import NumTok
from .tabfact_model import TabFactModel
from utils import setup_seed, time_str, WandbWrapper
from contextlib import nullcontext
from tqdm import tqdm
from datasets import Features, Sequence, ClassLabel, Value, Array2D
from transformers import TapasTokenizer, RobertaTokenizer


def load_args():
    parser_op = argparse.ArgumentParser("tabfact training.")
    # parser_op.add_argument('--nprocs', default=torch.cuda.device_count(), type=int, metavar='N')
    parser_op.add_argument('--apex', default=False, dest='apex', action='store_true')
    parser_op.add_argument("--run_name", type=str, default='tabfact', help='wandb runname')
    parser_op.add_argument("--encoder", type=str, default='tapas', choices=['tapas', 'roberta', 'bert'])
    parser_op.add_argument("--weights_path", type=str, default='', help='the path to pretrained checkpoint')
    parser_op.add_argument("--ckpt", type=str, default='',
                           help='The path to the finetuned checkpoint chosen to be validated/tested.'
                                'Required when mode is valid or test')
    # parser_op.add_argument("--mode", type=str, default='train', choices=['train', 'valid', 'test'])
    parser_op.add_argument('--use_numtok', type=int, default=1, help='0 for no use, 1 for replace, 2 for exnum, 3 for prompt')
    parser_op.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate.")
    parser_op.add_argument('--batch_size', type=int, default=48, help="batch size.")
    parser_op.add_argument('--gradient_accumulation_steps', type=int, default=1,
                           help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser_op.add_argument('--max_seq_len', type=int, default=512)
    parser_op.add_argument('--max_prompt_len', type=int, default=50)
    parser_op.add_argument("--max_epoch", type=int, default=10)
    parser_op.add_argument("--model_name", type=str, default='',
                           help="if empty(default),default model will be used;'zero' to use zero numbed")
    parser_op.add_argument("--checkpoint_path", type=str, default='',
                           help="The path to numbed checkpoint; If empty(default), random numbed will be used")
    parser_op.add_argument("--kept_keys", type=str, default='', help="param of Numtok.tokenize")  # TODO: help and logic
    parser_op.add_argument("--output_dir", type=str, default='/storage/tuna-models/tabfact-ckpt/')
    parser_op.add_argument("--data_dir", type=str, default='/storage/tuna-data/huggingface_TabFact/')
    parser_op.add_argument("--model_dir", type=str, default='/storage/tuna-models/')
    parser_op.add_argument("--redirect_huggingface_cache", type=int, default=1,
                           help='When set to 1, redirect the cache for GCR usage')
    parser_op.add_argument("--seed", type=int, default=42)
    # parser_op.add_argument("--log_name", type=str, default=f'result{time_str()}.log')
    parser_op.add_argument("--filter_numeric", action='store_true',
                           help='When set true, only considers questions that have number in them')
    parser_op.add_argument("--master_port", type=str, default='29500')
    parser_op.add_argument("--master_addr", type=str, default='localhost')
    args = parser_op.parse_args()
    assert args.max_seq_len <= 512

    if args.model_name == '':
        args.model_name = 'CharLSTM_base' if args.encoder in {'tapas', 'bert'} else 'CharLSTM_large'
    return args


def main(args):
    rank=int(os.environ["RANK"])
    args.rank = rank
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    setup_seed(args.seed + rank)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    logger.info("environments setting!")
    device = "cuda"
    print("rank:", rank)
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend=dist.Backend.NCCL, init_method="env://", world_size=args.world_size, rank=rank)
    dist.barrier()
    writer = WandbWrapper(args.rank == 0)
    # import wandb
    writer.init(name=args.run_name, config=args, notes=os.environ.get('WANDB_RUN_NOTES', None))
    writer.config.update({'aml_user': os.environ.get("USER", None),
                          'exp_name': os.environ.get("EXP_NAME", None),
                          'commit_hash': os.environ.get("COMMIT_HASH", None),
                          'cluster': os.environ.get("CLUSTER_NAME", None),
                          'git_branch': os.environ.get("GIT_BRANCH", None)
                          })
    args.device = device

    logger.info(f"load datasets.")

    train_dataset = load_dataset('tab_fact', 'tab_fact', split='train', cache_dir='/storage/tuna-data/huggingface_Data/' if args.redirect_huggingface_cache else None)
    valid_dataset = load_dataset('tab_fact', 'tab_fact', split='validation', cache_dir='/storage/tuna-data/huggingface_Data/' if args.redirect_huggingface_cache else None)
    test_dataset = load_dataset('tab_fact', 'tab_fact', split='test', cache_dir='/storage/tuna-data/huggingface_Data/' if args.redirect_huggingface_cache else None)

    if args.encoder in {'tapas', 'bert'}:
        tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact",
                                                   cache_dir=os.path.join(args.model_dir,
                                                                          'huggingface') if args.redirect_huggingface_cache else None)
    elif args.encoder == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(os.path.join(args.model_dir,'roberta.large'))

    tokenizer.add_tokens(["[NUM]"])
    tokenizer.num_token_id = len(tokenizer) - 1

    # we need to define the features ourselves as the token_type_ids of TAPAS are different from those of BERT
    features = Features({
        'attention_mask': Sequence(Value(dtype='int64')),
        'id': Value(dtype='int32'),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'label': ClassLabel(names=['refuted', 'entailed']),
        'statement': Value(dtype='string'),
        'table_caption': Value(dtype='string'),
        'table_id': Value(dtype='string'),
        'table_text': Value(dtype='string'),
        'token_type_ids': Array2D(dtype="int64",
                                  shape=(args.max_seq_len, 7 if args.encoder in {'tapas', 'bert'} else 1)),
        'number_list': Value(dtype='string'),
        'has_number': Value(dtype='int32')
    })

    def get_tapas_sample(e):
        has_number = 1 if NumTok.find_numbers(e['statement']) else 0
        if args.use_numtok == 1:
            stat_text, stat_number_list = NumTok.replace_numbers(e['statement'], do_lower_case=True)
            table_text, table_number_list = NumTok.replace_numbers(e['table_text'], do_lower_case=True)  # 替换number
            number_list = stat_number_list + table_number_list
        elif args.use_numtok >= 2:
            stat_text, stat_number_list = NumTok.replace_numbers(e['statement'], do_lower_case=True, keep_origin=1)
            table_text, table_number_list = NumTok.replace_numbers(e['table_text'], do_lower_case=True,
                                                                   keep_origin=1)  # 替换number
            number_list = stat_number_list + table_number_list
        else:
            table_text = e['table_text']
            stat_text = e['statement']
            number_list = []

        tk = tokenizer(table=read_text_as_pandas_table(table_text), queries=stat_text, padding='max_length')
        for k, v in tk.items():
            if len(v) > args.max_seq_len:
                tk[k] = v[:args.max_seq_len]

        tk['number_list'] = '#'.join(
            [x[0] for x in number_list[:sum([1 for i in tk['input_ids'] if i == tokenizer.num_token_id])]])
        tk['label'] = e['label']
        tk['has_number'] = has_number
        return tk

    def get_roberta_sample(e):
        tk = {}
        has_number = 1 if NumTok.find_numbers(e['statement']) else 0
        stat_ids, stat_number_strings = roberta_string_tokenize(e['statement'], tokenizer, args.use_numtok)
        table_ids, table_number_strings = roberta_table_tokenize(e['table_text'], tokenizer, args.use_numtok)
        ids = [tokenizer.cls_token_id] + stat_ids + [tokenizer.sep_token_id] + table_ids + [tokenizer.sep_token_id]
        number_strings = stat_number_strings + table_number_strings
        if len(ids) > args.max_seq_len:
            mask = [1] * args.max_seq_len
            ids = ids[:args.max_seq_len]
            number_list = '#'.join(number_strings[:sum([1 for i in ids if i == tokenizer.num_token_id])])
        else:
            mask = [1] * len(ids) + [0] * (args.max_seq_len - len(ids))
            number_list = '#'.join(number_strings)
            ids = ids + [tokenizer.pad_token_id] * (args.max_seq_len - len(ids))
        tk['input_ids'] = ids
        tk['attention_mask'] = mask
        tk['token_type_ids'] = [[0]] * args.max_seq_len
        tk['number_list'] = number_list
        tk['label'] = e['label']
        tk['has_number'] = has_number
        return tk

    dataset_numtok_suffixes = {
        0: '_ori',
        1: '_num',
        2: '_both',
        3: '_both',
    }
    train = train_dataset.map(
        get_tapas_sample if args.encoder in {'tapas', 'bert'} else get_roberta_sample,
        features=features,
        num_proc=16,
        cache_file_name=os.path.join(args.data_dir,
                                     f"train_tab_{'roberta' if args.encoder == 'roberta' else 'tapas'}{dataset_numtok_suffixes[args.use_numtok]}.cache")
    )
    valid = valid_dataset.map(
        get_tapas_sample if args.encoder in {'tapas', 'bert'} else get_roberta_sample,
        features=features,
        num_proc=16,
        cache_file_name=os.path.join(args.data_dir,
                                     f"valid_tab_{'roberta' if args.encoder == 'roberta' else 'tapas'}{dataset_numtok_suffixes[args.use_numtok]}.cache")
    )
    test = test_dataset.map(
        get_tapas_sample if args.encoder in {'tapas', 'bert'} else get_roberta_sample,
        features=features,
        num_proc=16,
        cache_file_name=os.path.join(args.data_dir,
                                     f"test_tab_{'roberta' if args.encoder == 'roberta' else 'tapas'}{dataset_numtok_suffixes[args.use_numtok]}.cache")
    )

    # map to PyTorch tensors and only keep columns we need
    train.set_format(columns=['input_ids', 'attention_mask', 'token_type_ids', 'number_list', 'label'])
    valid.set_format(columns=['input_ids', 'attention_mask', 'token_type_ids', 'number_list', 'label', 'has_number', 'table_id'])
    test.set_format(columns=['input_ids', 'attention_mask', 'token_type_ids', 'number_list', 'label', 'has_number', 'table_id'])

    # create PyTorch dataloader
    train, train_dataloader, train_sample = prepare_data(train, True, args)
    valid, valid_dataloader, valid_sample = prepare_data(valid, False, args)
    test, test_dataloader, test_sample = prepare_data(test, False, args)
    logger.info("Starting finetuning...")

    tabfactModel = TabFactModel(args, tokenizer).to(device)
    tabfactModel = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tabfactModel)

    if args.weights_path != '':
        weights_dict = torch.load(args.weights_path, map_location='cpu')['model']
        for k, v in list(weights_dict.items()):
            if k.split('.', 1)[0] not in {'number_model', 'encoder'}:
                del weights_dict[k]
        print(tabfactModel.load_state_dict(weights_dict, strict=False))

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    split_set = {}
    for key in ['complex']:
        with open(os.path.join(args.data_dir, '%s.json' % key)) as f:
            split_set[key] = set(json.load(f))

    optimizer = AdamW(tabfactModel.parameters(), lr=args.learning_rate)
    ddp_model = DDP(tabfactModel, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
    for epoch in range(0, args.max_epoch):
        train_sample.set_epoch((epoch))
        logger.info('At epoch {}'.format(epoch))
        ddp_model.train()
        for step, batch in enumerate(train_dataloader):
            # if step>10:break
            batch['input_ids'] = torch.stack(batch["input_ids"]).T.contiguous().to(
                device)  # [batch_size, hidden_size]
            batch['attention_mask'] = torch.stack(batch["attention_mask"]).T.contiguous().to(device)
            batch['token_type_ids'] = torch.stack([torch.stack(x) for x in batch["token_type_ids"]]).permute(2, 0,
                                                                                                             1).contiguous().to(
                device)
            batch['label'] = batch["label"].to(device)
            length = torch.where(batch['input_ids'] != 0)[1].max() + 1
            for k in ['input_ids', 'attention_mask', 'token_type_ids']:
                batch[k] = batch[k][:, :length]
            if args.encoder == 'roberta':
                del batch['token_type_ids']
            elif args.encoder == 'bert':
                batch['token_type_ids'] = batch['token_type_ids'][..., 0]

            mcontext = ddp_model.no_sync if step % args.gradient_accumulation_steps != 0 else nullcontext
            with mcontext():
                loss, acc, _ = ddp_model(**batch)
                agg_loss = loss / args.gradient_accumulation_steps
                agg_loss.backward()
            # 轮数为accumulation_steps整数倍的时候，传播梯度，并更新参数
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step % 20 == 0 and rank == 0:
                # logger.info(
                #     'epoch:%d,step:%d/%d,loss:%.4f,acc:%.4f' % (epoch, step, total_step, loss.item(), acc.item()))
                msg = "Epoch:%d" % epoch
                log = {}
                msg += ', %s:%.6f' % ('loss', loss.item())
                msg += ', %s:%.6f' % ('acc', acc.item())
                log['loss'] = loss.item()
                log['acc'] = acc.item()
                log = {'train': log}
                logger.info(msg)
                writer.log(log)
        # if rank == 0:
            # torch.save(ddp_model.module.state_dict(), os.path.join(args.output_dir, "net_%d.pt" % epoch))

        # evaluate

        ddp_model.eval()
        with torch.no_grad():
            for phase, loader in zip(['valid','test'],[valid_dataloader,test_dataloader]):
                accuracy = {
                key: load_metric("accuracy", cache_dir=f'.cache/{epoch}/{phase}/{key}', process_id=args.rank, num_process=args.world_size)
                for key in ['complex', 'all']}

                for dev_step,batch in enumerate(tqdm(loader)):
                    # get the inputs
                    # if dev_step > 10: break
                    batch['input_ids'] = torch.stack(batch["input_ids"]).T.contiguous().to(
                        device)  # [batch_size, hidden_size]
                    batch['attention_mask'] = torch.stack(batch["attention_mask"]).T.contiguous().to(device)
                    batch['token_type_ids'] = torch.stack([torch.stack(x) for x in batch["token_type_ids"]]).permute(2,0,1).contiguous().to(device)
                    batch['label'] = batch["label"].to(device)
                    batch['has_number'] = (batch['has_number'] == 1).to(device)
                    length = torch.where(batch['input_ids'] != 0)[1].max() + 1
                    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
                        batch[k] = batch[k][:, :length]
                    if args.encoder == 'roberta':
                        del batch['token_type_ids']
                    elif args.encoder == 'bert':
                        batch['token_type_ids'] = batch['token_type_ids'][..., 0]
                    loss, acc, prediction = ddp_model(**batch)
                    # add metric
                    for metric in [accuracy]:
                        for key in ['complex']:
                            slice = [table_id in split_set[key] for table_id in batch['table_id']]
                            metric[key].add_batch(predictions=prediction[slice], references=batch['label'][slice])
                        metric['all'].add_batch(predictions=prediction, references=batch['label'])

                log={}
                for key in ['complex', 'all']:
                    final_accuracy_score = accuracy[key].compute()
                    if writer.is_main_process:
                        logger.info("*** %s ***" % key)
                        logger.info("final_accuracy_score: %.6f" % final_accuracy_score['accuracy'])
                        log[key]={'acc':final_accuracy_score['accuracy']}

                if writer.is_main_process:
                    log = {phase:log, 'epoch': epoch}
                    writer.log(log)

    # elif args.mode in {'valid','test'}:
    #     assert args.nprocs == 1
    #     assert args.gradient_accumulation_steps == 1
    #     assert args.ckpt != ''
    #     print('Testing! Using filter_numeric', args.filter_numeric)
    #     tabfactModel.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    #     ddp_model = DDP(tabfactModel, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)



if __name__ == '__main__':
    args = load_args()
    # mp.spawn(main, args=(args,), nprocs=args.nprocs)
    main(args)
