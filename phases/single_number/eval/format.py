import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from phases.single_number.data.format_dataset import FormatDataset, collate
from number_encoder.Format.Format_model import Format
from number_encoder.Format.Format_config import Format_Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_suffix', default='')
    parser.add_argument('--preprocess_type', default='')
    parser.add_argument('--mode', default='')
    parser.add_argument('--eval_num_class', type=int, default=10)
    parser.add_argument('--comment', default='default_comment')
    parser.add_argument('--model_checkpoint_path', type=str, default='')
    parser.add_argument('--epoch_num', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--emb_size', type=int, default=1024)
    parser.add_argument('--lstm_num_layers', type=int, default=3)
    parser.add_argument('--dataset_name', default='unique')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    return args


def eval(args):
    print('\n\n\n')
    print(args)
    print('\n')

    seed = args.seed
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    lr = args.lr
    preprocess_type = args.preprocess_type
    epoch_num = args.epoch_num

    device = torch.device("cuda:{}".format(args.device)
                          if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    NLL_loss_func = nn.NLLLoss()
    NLL_loss_func = NLL_loss_func.to(device)

    config = Format_Config(args)
    model = Format(config)
    model.to(device)

    train_dataset = FormatDataset(dataset_name, is_train=True,
                                  preprocess_type=preprocess_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=16, pin_memory=True, shuffle=True,
                              collate_fn=collate)

    valid_dataset = FormatDataset(dataset_name, is_train=False,
                                  preprocess_type=preprocess_type)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                              num_workers=16, pin_memory=True, shuffle=True,
                              collate_fn=collate)
    print('\n')
    print('Using {} training samples'.format(len(train_dataset)))
    print('Using {} testing samples'.format(len(valid_dataset)))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=lr)

    best_epoch = 0
    best_frac = (0, 0)
    best_in01 = (0, 0)
    best_in0100 = (0, 0)
    tic = time.time()
    for epoch in range(epoch_num):

        # Early Stop
        if epoch - best_epoch > 20:
            print('Early stopped!')
            break

        # Train the Evaluation Model on training set ##
        total_train_loss = []
        model.train()
        for (token_ids, seq_len, sig_gt, exp_gt, val_gt, number_string,
             batch_frac_digit, batch_in01, batch_in0100) in tqdm(train_loader):

            token_ids = token_ids.to(device)
            sig_gt = sig_gt.to(device)
            seq_len = seq_len.to(device)
            exp_gt = exp_gt.to(device)
            val_gt = val_gt.to(device)
            batch_frac_digit = batch_frac_digit.to(device)
            batch_in01 = batch_in01.to(device)
            batch_in0100 = batch_in0100.to(device)
            input_ids = number_string['input_ids'].to(device)
            attention_mask = number_string['attention_mask'].to(device)

            inputs = (token_ids, seq_len, sig_gt,
                      exp_gt, val_gt, input_ids, attention_mask)
            frac_digit_pred, in01_pred, in0100_pred = model(inputs)

            loss1 = NLL_loss_func(frac_digit_pred, batch_frac_digit)
            loss2 = NLL_loss_func(in01_pred, batch_in01)
            loss3 = NLL_loss_func(in0100_pred, batch_in0100)

            optimizer.zero_grad()
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()

            total_train_loss.append(loss1.item() + loss2.item() + loss3.item())
        current_train_loss = np.mean(total_train_loss)

        # Eval on validation set ##
        model.eval()
        correct_frac_digit_cnt = 0
        correct_in01_cnt = 0
        correct_in0100_cnt = 0
        for (token_ids, seq_len, sig_gt, exp_gt, val_gt, number_string,
             batch_frac_digit, batch_in01, batch_in0100) in tqdm(valid_loader):

            token_ids = token_ids.to(device)
            seq_len = seq_len.to(device)
            sig_gt = sig_gt.to(device)
            exp_gt = exp_gt.to(device)
            val_gt = val_gt.to(device)
            batch_frac_digit = batch_frac_digit.to(device)
            batch_in01 = batch_in01.to(device)
            batch_in0100 = batch_in0100.to(device)
            input_ids = number_string['input_ids'].to(device)
            attention_mask = number_string['attention_mask'].to(device)

            with torch.no_grad():
                inputs = (token_ids, seq_len, sig_gt, exp_gt,
                          val_gt, input_ids, attention_mask)
                frac_digit_pred, in01_pred, in0100_pred = model(inputs)
                frac_digit_indexed = torch.argmax(frac_digit_pred, dim=1)
                in01_indexed = torch.argmax(in01_pred)
                in0100_indexed = torch.argmax(in0100_pred)

                results = (frac_digit_indexed == batch_frac_digit)
                correct_frac_digit_cnt += torch.sum(results)
                correct_in01_cnt += torch.sum(in01_indexed == batch_in01)
                correct_in0100_cnt += torch.sum(in0100_indexed == batch_in0100)

        current_frac_accuracy = correct_frac_digit_cnt / len(valid_dataset)
        current_in01_accuracy = correct_in01_cnt / len(valid_dataset)
        current_in0100_accuracy = correct_in0100_cnt / len(valid_dataset)

        # update best valid
        if current_frac_accuracy > best_frac[1]:
            best_frac = (epoch, current_frac_accuracy)
            best_epoch = epoch

        if current_in01_accuracy > best_in01[1]:
            best_in01 = (epoch, current_in01_accuracy)
            best_epoch = epoch

        if current_in0100_accuracy > best_in0100[1]:
            best_in0100 = (epoch, current_in0100_accuracy)
            best_epoch = epoch

        toc = time.time()

        print('Epoch: {} \t train NLL loss sum: {} \t Time elapsed: {} '
              .format(epoch, current_train_loss, toc - tic))
        print('[Validation] \t frac Acc: {} \t in01 Acc: {} \t in0100 Acc: {}'
              .format(
                current_frac_accuracy,
                current_in01_accuracy,
                current_in0100_accuracy
                ))
        print('[Bests] \t frac Acc: {} \t in01 Acc: {} \t in0100 Acc: {}'
              .format(
                best_frac,
                best_in01,
                best_in0100
                ))


if __name__ == '__main__':
    args = parse_args()
    eval(args)
