import logging
import os
from os import path

import torch
import transformers
from accelerate import Accelerator
from number_encoder import NumBedConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .args import define_main_parser
from .dataset.card import TransactionDataset
from .dataset.datacollator import numtok_collate_fn
from .misc.utils import random_split_dataset
from .models.modules import TabFormerBertLM

logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

accelerator = Accelerator()
logger.info(accelerator.state)
logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()


def get_args():
    parser = define_main_parser()
    parser.add_argument('--model_name_or_path', type=str, default=None, help='directory of the TabFormer model')
    parser.add_argument('--cached_feature_dir', type=str, default=None, help='directory of the cached feature')

    args = parser.parse_args()

    return args


def extract_features(args):
    """Extracts features for each sample using pre-trained TabBERT.

    The extracted features will be stored in args.cached_feature_dir for further use.
    """

    if args.data_type == 'card':
        dataset = TransactionDataset(root=args.data_root,
                                     fname=args.data_fname,
                                     fextension=args.data_extension,
                                     vocab_dir=args.output_dir,
                                     nrows=args.nrows,
                                     user_ids=args.user_ids,
                                     mlm=args.mlm,
                                     cached=args.cached,
                                     stride=10,  # Follows the description in the paper.
                                     flatten=args.flatten,
                                     return_labels=True,
                                     skip_user=args.skip_user,
                                     use_numtok=args.use_numtok,
                                     log=logger)
    else:
        raise NotImplementedError(f"data type {args.data_type} not defined")

    os.makedirs(args.cached_feature_dir, exist_ok=True)

    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    # split dataset into train, val, test [0.6. 0.2, 0.2]
    totalN = len(dataset)
    trainN = int(0.6 * totalN)

    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN

    assert totalN == trainN + valN + testN

    lengths = [trainN, valN, testN]

    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))
    train_dataset, valid_dataset, test_dataset = random_split_dataset(dataset, lengths)

    if args.use_numtok:
        number_model_config = NumBedConfig(model_name=args.number_model_config)
    else:
        number_model_config = None

    if args.lm_type == "bert":
        tab_net = TabFormerBertLM(custom_special_tokens,
                                  vocab=vocab,
                                  field_ce=args.field_ce,
                                  flatten=args.flatten,
                                  ncols=dataset.ncols,
                                  field_hidden_size=args.field_hs,
                                  use_numtok=args.use_numtok,
                                  number_model_config=number_model_config,
                                  use_replace=args.use_replace,
                                  data_type='card'
                                  )
    else:
        raise NotImplementedError('Currently only support bert.')

    log.info(f'Load model from {args.model_name_or_path}, use numtok = {args.use_numtok}.')
    tab_net.model.load_state_dict(
        torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))

    batch_size = args.per_device_eval_batch_size

    if args.use_numtok:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=numtok_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=numtok_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=numtok_collate_fn)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    tab_net.model, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        tab_net.model, train_dataloader, valid_dataloader, test_dataloader
    )

    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []
    test_features = []
    test_labels = []

    tab_net.model.eval()
    accelerator.wait_for_everyone()
    logger.info('Extracting train features.')
    for data in tqdm(train_dataloader):
        with torch.no_grad():
            if args.use_numtok:
                sequence_output = tab_net.model(input_ids=data['input_ids'], tokenized_number=data['tokenized_numbers'],
                                                mlm=False)
            else:
                sequence_output = tab_net.model(input_ids=data['input_ids'])  # [bsz * seqlen * hidden]
        train_features.append(accelerator.gather(sequence_output).cpu())
        train_labels.append(accelerator.gather(data['labels']).cpu())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f'save training features and labels to {args.cached_feature_dir}')
        train_features = torch.cat(train_features)
        train_labels = torch.cat(train_labels)

        train_feature_fname = path.join(args.cached_feature_dir, 'train_feature.pth')
        train_label_fname = path.join(args.cached_feature_dir, 'train_label.pth')

        torch.save(train_features, train_feature_fname)
        torch.save(train_labels, train_label_fname)

    accelerator.wait_for_everyone()
    logger.info('Extracting valid features.')
    for data in tqdm(valid_dataloader):
        with torch.no_grad():
            if args.use_numtok:
                sequence_output = tab_net.model(input_ids=data['input_ids'], tokenized_number=data['tokenized_numbers'],
                                                mlm=False)
            else:
                sequence_output = tab_net.model(input_ids=data['input_ids'])  # [bsz * seqlen * hidden]
        valid_features.append(accelerator.gather(sequence_output).cpu())
        valid_labels.append(accelerator.gather(data['labels']).cpu())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f'save valid features and labels to {args.cached_feature_dir}')
        valid_features = torch.cat(valid_features)
        valid_labels = torch.cat(valid_labels)

        valid_feature_fname = path.join(args.cached_feature_dir, 'valid_feature.pth')
        valid_label_fname = path.join(args.cached_feature_dir, 'valid_label.pth')

        torch.save(valid_features, valid_feature_fname)
        torch.save(valid_labels, valid_label_fname)

    accelerator.wait_for_everyone()
    logger.info('Extracting test features.')
    for data in tqdm(test_dataloader):
        with torch.no_grad():
            if args.use_numtok:
                sequence_output = tab_net.model(input_ids=data['input_ids'], tokenized_number=data['tokenized_numbers'],
                                                mlm=False)
            else:
                sequence_output = tab_net.model(input_ids=data['input_ids'])  # [bsz * seqlen * hidden]
        test_features.append(accelerator.gather(sequence_output).cpu())
        test_labels.append(accelerator.gather(data['labels']).cpu())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f'save test features and labels to {args.cached_feature_dir}')
        test_features = torch.cat(test_features)
        test_labels = torch.cat(test_labels)

        test_feature_fname = path.join(args.cached_feature_dir, 'test_feature.pth')
        test_label_fname = path.join(args.cached_feature_dir, 'test_label.pth')

        torch.save(test_features, test_feature_fname)
        torch.save(test_labels, test_label_fname)


def main():
    args = get_args()
    extract_features(args)


if __name__ == '__main__':
    main()
