import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from data.dataset.listmax_dataset import ListMaxDataset, collate
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabular_model.number_encoder import ListMax
from tabular_model.number_encoder import ListMax_Config


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

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    MSE_loss_func = nn.MSELoss()
    MSE_loss_func = MSE_loss_func.to(device)
    NLL_loss_func = nn.NLLLoss()
    NLL_loss_func = NLL_loss_func.to(device)

    train_dataset = ListMaxDataset(dataset_name, is_train=True, preprocess_type=preprocess_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=True,
                              collate_fn=collate)
    valid_dataset = ListMaxDataset(dataset_name, is_train=False, preprocess_type=preprocess_type)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=True,
                              collate_fn=collate)
    print('\n')
    print('Using {} training samples'.format(len(train_dataset)))
    print('Using {} testing samples'.format(len(valid_dataset)))

    config = ListMax_Config(args)
    model = ListMax(config)
    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_epoch = 0
    best_max_id_acc = (0, 0)
    tic = time.time()
    for epoch in range(epoch_num):

        # Early Stop
        if epoch - best_epoch > 20:
            print('Early stopped!')
            break

        total_train_loss = []
        model.train()

        for (numbers, results) in tqdm(train_loader):
            inputs = []
            for number in numbers:
                batch_token_ids, seq_len, sig_gt, exp_gt, batch_val, batch_number_string = number
                batch_token_ids = batch_token_ids.to(device)
                sig_gt = sig_gt.to(device)
                seq_len = seq_len.to(device)
                exp_gt = exp_gt.to(device)
                batch_val = batch_val.to(device)
                input_ids = batch_number_string['input_ids'].to(device)
                attention_mask = batch_number_string['attention_mask'].to(device)
                inputs.append((batch_token_ids, seq_len, sig_gt, exp_gt, batch_val, input_ids, attention_mask))

            max_id_pred = model(inputs)
            max_id_gt = results[0]
            max_id_gt = max_id_gt.to(device)

            loss = NLL_loss_func(max_id_pred, max_id_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss.append(loss.item())
        current_train_loss = np.mean(total_train_loss)

        ## Eval on validation set ##
        model.eval()
        correct_cnt = 0
        for (numbers, results) in tqdm(valid_loader):
            inputs = []
            for number in numbers:
                batch_token_ids, seq_len, sig_gt, exp_gt, batch_val, batch_number_string = number
                batch_token_ids = batch_token_ids.to(device)
                sig_gt = sig_gt.to(device)
                seq_len = seq_len.to(device)
                exp_gt = exp_gt.to(device)
                batch_val = batch_val.to(device)
                input_ids = batch_number_string['input_ids'].to(device)
                attention_mask = batch_number_string['attention_mask'].to(device)
                inputs.append((batch_token_ids, seq_len, sig_gt, exp_gt, batch_val, input_ids, attention_mask))

            with torch.no_grad():
                max_id_pred = model(inputs)
                max_id_gt = results[0]
                max_id_gt = max_id_gt.to(device)
                max_id_indexed = torch.argmax(max_id_pred, dim=1)

                correct_cnt += torch.sum(max_id_indexed == max_id_gt)

        current_max_id_accuracy = correct_cnt / len(valid_dataset)

        # update best valid
        if current_max_id_accuracy > best_max_id_acc[1]:
            best_max_id_acc = (epoch, current_max_id_accuracy)
            best_epoch = epoch

        toc = time.time()
        print('Epoch: {} \t train NLL loss: {} \t Time elapsed: {} '.format(epoch, math.sqrt(current_train_loss),
                                                                            toc - tic))
        print('[Validation] \t max_id Acc: {}'.format(current_max_id_accuracy))
        print('[Bests] \t exp Acc: {}'.format((best_max_id_acc[0], best_max_id_acc[1].item())))

    return best_max_id_acc[1]


if __name__ == '__main__':
    args = parse_args()
    eval(args)
