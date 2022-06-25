import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from phases.single_number.data.decode_dataset import DecodeDataset, collate
from number_encoder.Evaluation.Evaluation_model import Evaluation
from number_encoder.Evaluation.Evaluation_config import Evaluation_Config


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
    MSE_loss_func = nn.MSELoss()
    MSE_loss_func = MSE_loss_func.to(device)
    NLL_loss_func = nn.NLLLoss()
    NLL_loss_func = NLL_loss_func.to(device)

    config = Evaluation_Config(args)
    model = Evaluation(config)
    model.to(device)

    train_dataset = DecodeDataset(dataset_name, is_train=True,
                                  preprocess_type=preprocess_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=16, pin_memory=True, shuffle=True,
                              collate_fn=collate)
    valid_dataset = DecodeDataset(dataset_name, is_train=False,
                                  preprocess_type=preprocess_type)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=16, pin_memory=True, shuffle=True,
                              collate_fn=collate)
    print('\n')
    print('Using {} training samples'.format(len(train_dataset)))
    print('Using {} testing samples'.format(len(valid_dataset)))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        model.parameters()), lr=lr)

    best_valid_loss = np.inf
    best_epoch = 0
    best_sig = (0, np.inf)
    best_val = (0, np.inf)
    best_exp = (0, 0)
    tic = time.time()
    for epoch in range(epoch_num):

        # Early Stop
        if epoch - best_epoch > 20:
            print('Early stopped!')
            break

        # Train the Evaluation Model on training set ##
        total_train_loss = []
        model.train()
        for (token_ids, seq_len, sig_gt, exp_gt, val_gt, number_string) in tqdm(train_loader):
            token_ids = token_ids.to(device)
            sig_gt = sig_gt.to(device)
            seq_len = seq_len.to(device)
            exp_gt = exp_gt.to(device)
            val_gt = val_gt.to(device)
            input_ids = number_string['input_ids'].to(device)
            attention_mask = number_string['attention_mask'].to(device)

            sig_pred, exp_pred = model(token_ids, seq_len, sig_gt, exp_gt, val_gt, input_ids, attention_mask)

            loss1 = MSE_loss_func(sig_pred.view(sig_pred.size(0)), sig_gt.view(sig_gt.size(0)))
            loss2 = NLL_loss_func(exp_pred, exp_gt)

            optimizer.zero_grad()
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_train_loss.append(loss1.item())
        current_train_loss = np.mean(total_train_loss)

        # Eval on validation set ##
        total_valid_sig_mse = []
        total_valid_val_mse = []
        model.eval()
        correct_cnt = 0
        for (token_ids, seq_len, sig_gt, exp_gt, val_gt, number_string) in tqdm(valid_loader):
            token_ids = token_ids.to(device)
            seq_len = seq_len.to(device)
            sig_gt = sig_gt.to(device)
            exp_gt = exp_gt.to(device)
            val_gt = val_gt.to(device)
            input_ids = number_string['input_ids'].to(device)
            attention_mask = number_string['attention_mask'].to(device)

            with torch.no_grad():
                sig_pred, exp_pred = model(token_ids, seq_len, sig_gt, exp_gt, val_gt, input_ids, attention_mask)
                exp_indexed = torch.argmax(exp_pred, dim=1)
                sig_pred = sig_pred.view(sig_pred.size(0))
                val_pred = sig_pred * 10 ** (exp_indexed - 2)

                loss_sig = MSE_loss_func(sig_pred.view(sig_pred.size(0)), sig_gt.view(sig_gt.size(0)))
                loss_val = MSE_loss_func(val_pred, val_gt)
                correct_cnt += torch.sum(exp_indexed == exp_gt)
                total_valid_sig_mse.append(loss_sig.item())
                total_valid_val_mse.append(loss_val.item())

        current_valid_sig_loss = np.mean(total_valid_sig_mse)
        current_valid_val_loss = np.mean(total_valid_val_mse)
        current_exp_accuracy = correct_cnt / len(valid_dataset)

        # update best valid
        if current_valid_sig_loss < best_valid_loss:
            best_valid_loss = current_valid_sig_loss

        # update bests
        if current_valid_sig_loss < best_sig[1]:
            best_sig = (epoch, current_valid_sig_loss)
            best_epoch = epoch

        if current_valid_val_loss < best_val[1]:
            best_val = (epoch, current_valid_val_loss)
            best_epoch = epoch

        if current_exp_accuracy > best_exp[1]:
            best_exp = (epoch, current_exp_accuracy)
            best_epoch = epoch

        toc = time.time()

        print('Epoch: {} \t train sig RMSE: {} \t Time elapsed: {} '.format(epoch, math.sqrt(current_train_loss),
                                                                            toc - tic))
        print('[Validation] \t sig RMSE: {} \t exp Acc: {} \t val RMSE: {}'.format(
            math.sqrt(current_valid_sig_loss),
            current_exp_accuracy,
            math.sqrt(current_valid_val_loss)
        ))
        print('[Bests] \t sig RMSE: {} \t exp Acc: {} \t val RMSE: {}'.format(
            (best_sig[0], math.sqrt(best_sig[1])),
            (best_exp[0], best_exp[1].item()),
            (best_val[0], math.sqrt(best_val[1]))
        ))
    return best_valid_loss


if __name__ == '__main__':
    args = parse_args()
    eval(args)
