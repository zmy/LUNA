# python -m phases.single_number.phase0_new

import argparse
import math
import os
from math import inf

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from number_encoder import NumBedConfig, MODEL_NAMES
from utils import to_device
from .data import DoubleDataset
from .data import MultiDataset
from .data import SingleDataset
from .phase0_model import Phase0, TripletLoss
from utils import num_params
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='CharLSTM',
                        choices=MODEL_NAMES)
    parser.add_argument("--encoder_name", type=str, default='TaPas')
    parser.add_argument('--model_suffix', default='Bi')
    parser.add_argument('--preprocess_type', default='trivial')
    parser.add_argument('--mode', default='sigexp')
    parser.add_argument('--value_ratio', type=float, default=0.25)
    parser.add_argument('--objs', nargs='*',
                        default=['single', 'double', 'multi'])
    parser.add_argument('--evals', nargs='*',
                        default=['single', 'double', 'multi'])
    parser.add_argument('--exp_root', default='exp')
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dataset_name', default='sampled')
    parser.add_argument('--listmax_dataset_name', default='sampled')
    parser.add_argument('--emb_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--lstm_num_layers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--direct_expand', action='store_true')
    parser.add_argument('--checkpoint_path', default='')
    parser.add_argument('--save_freq', type=int, default=3)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument("--world_size", default=3)
    parser.add_argument("--mix", default='cat')
    parser.add_argument("--aligned", action='store_true')
    parser.add_argument("--use_layer_norm", action='store_true')
    parser.add_argument("--align_with_orig", action='store_true')
    parser.add_argument("--tb_root_dir", type=str, default='/tb_logs/p0.5/')
    return parser.parse_args()


def main(args):
    print('\n\n\n')
    print(args)
    print('\n')

    seed = args.seed
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    lr = args.lr
    preprocess_type = args.preprocess_type
    epoch_num = args.epoch_num
    exp_root = args.exp_root
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    tb_dir = os.path.join(args.tb_root_dir, args.model_name)
    writer = SummaryWriter(tb_dir)
    print('Tensorboard written to:', tb_dir)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    MSE_loss_func = nn.MSELoss()
    MSE_loss_func = MSE_loss_func.to(device)
    NLL_loss_func = nn.NLLLoss()
    NLL_loss_func = NLL_loss_func.to(device)
    Triplet_loss_func = TripletLoss()
    Triplet_loss_func = Triplet_loss_func.to(device)

    config = NumBedConfig(model_name=args.model_name,
                          encoder_name=args.encoder_name)
    model = Phase0(config)
    model.to(device)
    print('\n\n')
    print('Current number of parameters:', num_params(model.core_model))

    model_save_dir = os.path.join(exp_root, 'checkpoints', 'phase0_pretrained', f'{args.model_name}')

    # Single dataset
    single_train_dataset = SingleDataset(dataset_name, is_train=True, kept_keys=model.core_model.param_keys)
    single_train_loader = DataLoader(single_train_dataset,
                                     batch_size=batch_size,
                                     num_workers=16,
                                     pin_memory=True,
                                     shuffle=True,
                                     collate_fn=single_train_dataset.single_collate)
    single_valid_dataset = SingleDataset(dataset_name, is_train=False, kept_keys=model.core_model.param_keys)
    single_valid_loader = DataLoader(single_valid_dataset,
                                     batch_size=batch_size,
                                     num_workers=16,
                                     pin_memory=True,
                                     shuffle=True,
                                     collate_fn=single_valid_dataset.single_collate)

    print('\n')
    print('Using {} single training samples'
          .format(len(single_train_dataset)))
    print('Using {} single testing samples'
          .format(len(single_valid_dataset)))

    # Double dataset
    double_train_dataset = DoubleDataset(dataset_name, is_train=True, kept_keys=model.core_model.param_keys,
                                         preprocess_type=preprocess_type)
    double_train_loader = DataLoader(double_train_dataset,
                                     batch_size=batch_size,
                                     num_workers=16,
                                     pin_memory=True,
                                     shuffle=True,
                                     collate_fn=double_train_dataset.double_collate)
    double_valid_dataset = DoubleDataset(dataset_name, is_train=False, kept_keys=model.core_model.param_keys,
                                         preprocess_type=preprocess_type)
    double_valid_loader = DataLoader(double_valid_dataset,
                                     batch_size=batch_size,
                                     num_workers=16,
                                     pin_memory=True,
                                     shuffle=True,
                                     collate_fn=double_valid_dataset.double_collate)

    print('\n')
    print('Using {} double training samples'
          .format(len(double_train_dataset)))
    print('Using {} double testing samples'
          .format(len(double_valid_dataset)))

    # Multi dataset
    multi_train_dataset = MultiDataset(dataset_name, is_train=True, kept_keys=model.core_model.param_keys,
                                       preprocess_type=preprocess_type)
    multi_train_loader = DataLoader(multi_train_dataset,
                                    batch_size=batch_size,
                                    num_workers=16,
                                    pin_memory=True,
                                    shuffle=True,
                                    collate_fn=multi_train_dataset.multi_collate)
    multi_valid_dataset = MultiDataset(dataset_name, is_train=False, kept_keys=model.core_model.param_keys,
                                       preprocess_type=preprocess_type)
    multi_valid_loader = DataLoader(multi_valid_dataset,
                                    batch_size=batch_size,
                                    num_workers=16,
                                    pin_memory=True,
                                    shuffle=True,
                                    collate_fn=multi_valid_dataset.multi_collate)
    print('\n')
    print('Using {} multi training samples'.format(len(multi_train_dataset)))
    print('Using {} multi testing samples'.format(len(multi_valid_dataset)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = (0, 0)
    best_rmse = (0, inf)
    for epoch in range(epoch_num):
        model.train()
        single_loss = 0
        double_loss = 0
        multi_loss = 0

        # TRAINING PHASE
        if args.objs:
            # single
            if 'single' in args.objs:
                flag = 1
                running_loss1 = 0
                running_loss2 = 0
                running_loss3 = 0
                running_loss4 = 0
                running_loss5 = 0
                running_loss6 = 0
                running_loss7 = 0
                for number_attr in tqdm(single_train_loader):
                    parameters = to_device(number_attr, device)
                    if flag:
                        print(parameters.keys())
                        flag = 0
                    output = model(task='single', input_dicts=[parameters])
                    sig_pred, exp_pred, log_pred, frac_pred, in01_pred, in0100_pred, feats = output
                    sig_gt = parameters['batch_sig']
                    exp_gt = parameters['batch_exp']
                    log_gt = parameters['batch_log']
                    frac_gt = parameters['batch_frac']
                    in01_gt = parameters['batch_in01']
                    in0100_gt = parameters['batch_in0100']

                    loss1 = MSE_loss_func(sig_pred.view(sig_pred.size(0)), sig_gt.view(sig_gt.size(0)))
                    loss2 = NLL_loss_func(exp_pred, exp_gt.view(exp_gt.size(0)))
                    loss3 = MSE_loss_func(log_pred.view(log_pred.size(0)), log_gt.view(log_gt.size(0)))
                    loss4 = NLL_loss_func(frac_pred, frac_gt.view(frac_gt.size(0)))
                    loss5 = NLL_loss_func(in01_pred, in01_gt.view(in01_gt.size(0)))
                    loss6 = NLL_loss_func(in0100_pred, in0100_gt.view(in0100_gt.size(0)))
                    loss7 = Triplet_loss_func(feats).to(device)

                    optimizer.zero_grad()
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
                    loss.backward()
                    optimizer.step()
                    single_loss += loss.item()

                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()
                    running_loss3 += loss3.item()
                    running_loss4 += loss4.item()
                    running_loss5 += loss5.item()
                    running_loss6 += loss6.item()
                    running_loss7 += loss7.item()

                writer.add_scalar('single/sig', running_loss1 / len(single_train_loader), epoch)
                writer.add_scalar('single/exp', running_loss2 / len(single_train_loader), epoch)
                writer.add_scalar('single/log', running_loss3 / len(single_train_loader), epoch)
                writer.add_scalar('single/frac', running_loss4 / len(single_train_loader), epoch)
                writer.add_scalar('single/in01', running_loss5 / len(single_train_loader), epoch)
                writer.add_scalar('single/in0100', running_loss6 / len(single_train_loader), epoch)
                writer.add_scalar('single/feats', running_loss7 / len(single_train_loader), epoch)

            # double
            if 'double' in args.objs:
                loss = 0
                running_loss1 = 0
                running_loss2 = 0
                running_loss3 = 0
                running_loss4 = 0
                running_loss5 = 0
                running_loss6 = 0
                running_loss7 = 0
                running_loss8 = 0
                running_loss9 = 0
                for numbers, results in tqdm(double_train_loader):
                    inputs = []
                    for number in numbers:
                        parameters = to_device(number, device)
                        inputs.append(parameters)

                    output = model(task='double', input_dicts=inputs)

                    # Add
                    add_sig_pred, add_exp_pred, add_log_pred = output[0], output[1], output[2]
                    add_val_gt, add_sig_gt, add_exp_gt, add_log_gt = results[0]
                    add_val_gt = add_val_gt.to(device)
                    add_sig_gt = add_sig_gt.to(device)
                    add_exp_gt = add_exp_gt.to(device)
                    add_log_gt = add_log_gt.to(device)

                    # Subtract
                    subtract_sig_pred, subtract_exp_pred, subtract_log_pred, = output[3], output[4], output[5]
                    subtract_val_gt, subtract_sig_gt, subtract_exp_gt, subtract_log_gt = results[1]
                    subtract_val_gt = subtract_val_gt.to(device)
                    subtract_sig_gt = subtract_sig_gt.to(device)
                    subtract_exp_gt = subtract_exp_gt.to(device)
                    subtract_log_gt = subtract_log_gt.to(device)

                    # CP, CS
                    cp_pred, cs_pred = output[6], output[7]
                    cp_gt, cs_gt = results[2]
                    cp_gt = cp_gt.to(device)
                    cs_gt = cs_gt.to(device)

                    # Alignment
                    feats = output[8]

                    loss1 = MSE_loss_func(add_sig_pred.view(add_sig_pred.size(0)), add_sig_gt.view(add_sig_gt.size(0)))
                    loss2 = NLL_loss_func(add_exp_pred, add_exp_gt)
                    loss3 = MSE_loss_func(add_log_pred.view(add_log_pred.size(0)), add_log_gt.view(add_log_gt.size(0)))

                    loss4 = MSE_loss_func(
                        subtract_sig_pred.view(subtract_sig_pred.size(0)),
                        subtract_sig_gt.view(subtract_sig_gt.size(0)))
                    loss5 = NLL_loss_func(subtract_exp_pred, subtract_exp_gt)
                    loss6 = MSE_loss_func(
                        subtract_sig_pred.view(subtract_sig_pred.size(0)),
                        subtract_sig_gt.view(subtract_sig_gt.size(0)))

                    loss7 = MSE_loss_func(cp_pred.view(cp_pred.size(0)), cp_gt.view(cp_gt.size(0)))
                    loss8 = MSE_loss_func(cs_pred.view(cs_pred.size(0)), cs_gt.view(cs_gt.size(0)))

                    loss9 = Triplet_loss_func(feats).to(device)

                    optimizer.zero_grad()
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9
                    loss.backward()
                    optimizer.step()
                    double_loss += loss.item()

                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()
                    running_loss3 += loss3.item()
                    running_loss4 += loss4.item()
                    running_loss5 += loss5.item()
                    running_loss6 += loss6.item()
                    running_loss7 += loss7.item()
                    running_loss8 += loss8.item()
                    running_loss9 += loss9.item()

                writer.add_scalar('double/sig', running_loss1 / len(double_train_loader), epoch)
                writer.add_scalar('double/exp', running_loss2 / len(double_train_loader), epoch)
                writer.add_scalar('double/log', running_loss3 / len(double_train_loader), epoch)
                writer.add_scalar('double/subtract_sig', running_loss4 / len(double_train_loader), epoch)
                writer.add_scalar('double/subtract_exp', running_loss5 / len(double_train_loader), epoch)
                writer.add_scalar('double/subtract_log', running_loss6 / len(double_train_loader), epoch)
                writer.add_scalar('double/cp', running_loss7 / len(double_train_loader), epoch)
                writer.add_scalar('double/cs', running_loss8 / len(double_train_loader), epoch)

            # mutli
            if 'multi' in args.objs:
                running_loss1 = 0
                for (numbers, results) in tqdm(multi_train_loader):
                    inputs = []
                    for number in numbers:
                        parameters = to_device(number, device)
                        inputs.append(parameters)

                    max_id_pred, feats = model('multi', inputs)
                    max_id_gt = results[0]
                    max_id_gt = max_id_gt.to(device)
                    loss1 = NLL_loss_func(max_id_pred, max_id_gt)
                    loss2 = Triplet_loss_func(feats).to(device)
                    loss = loss1 + loss2

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    multi_loss += loss.item()
                    running_loss1 += loss1.item()
                writer.add_scalar('multi/max', running_loss1 / len(multi_train_loader), epoch)

        # SAVE MODEL
        if (epoch + 1) % args.save_freq == 0:
            file_name = os.path.join(model_save_dir,
                                     'epoch{}.pt'.format(epoch))
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            print('Model saved to', model_save_dir)
            torch.save(model.core_model.model.state_dict(), file_name)

        # EVALUATION
        if (epoch + 1) % args.eval_freq == 0:
            print('Start Evaluation!')
            total_acc = []
            total_rmse = []

            # decoding + formatting
            if 'single' in args.evals:
                total_valid_sig_mse = []
                total_valid_log_mse = []
                model.eval()
                exp_correct_cnt = 0
                frac_correct_cnt = 0
                in01_correct_cnt = 0
                in0100_correct_cnt = 0
                for number in tqdm(single_valid_loader):
                    parameters = to_device(number, device)
                    with torch.no_grad():
                        output = model(task='single', input_dicts=[parameters])
                        sig_pred, exp_pred, log_pred, frac_pred, in01_pred, in0100_pred, feats = output

                        sig_gt = parameters['batch_sig']
                        exp_gt = parameters['batch_exp']
                        log_gt = parameters['batch_log']
                        frac_gt = parameters['batch_frac']
                        in01_gt = parameters['batch_in01']
                        in0100_gt = parameters['batch_in0100']

                        loss_sig = MSE_loss_func(sig_pred.view(sig_pred.size(0)), sig_gt.view(sig_gt.size(0)))

                        loss_log = MSE_loss_func(log_pred.view(log_pred.size(0)), log_gt.view(log_gt.size(0)))

                        exp_indexed = torch.argmax(exp_pred, dim=1)
                        exp_indexed = exp_indexed.view(-1, 1)
                        exp_gt = exp_gt.view(-1, 1)
                        exp_correct_cnt += torch.sum(exp_indexed == exp_gt)

                        frac_indexed = torch.argmax(frac_pred, dim=1)
                        frac_indexed = frac_indexed.view(-1, 1)
                        frac_gt = frac_gt.view(-1, 1)
                        frac_correct_cnt += torch.sum(frac_indexed == frac_gt)

                        in01_indexed = torch.argmax(in01_pred, dim=1)
                        in01_indexed = in01_indexed.view(-1, 1)
                        in01_gt = in01_gt.view(-1, 1)
                        in01_correct_cnt += torch.sum(in01_indexed == in01_gt)

                        in0100_indexed = torch.argmax(in0100_pred, dim=1)
                        in0100_indexed = in0100_indexed.view(-1, 1)
                        in0100_gt = in0100_gt.view(-1, 1)
                        in0100_correct_cnt += torch.sum(in0100_indexed == in0100_gt)

                        total_valid_sig_mse.append(loss_sig.item())
                        total_valid_log_mse.append(loss_log.item())
                current_valid_sig_loss = np.mean(total_valid_sig_mse)
                current_valid_log_loss = np.mean(total_valid_log_mse)

                current_exp_accuracy = exp_correct_cnt / len(single_valid_dataset)
                current_frac_accuracy = frac_correct_cnt / len(single_valid_dataset)
                current_in01_accuracy = in01_correct_cnt / len(single_valid_dataset)
                current_in0100_accuracy = in0100_correct_cnt / len(single_valid_dataset)

                print('valid_sig RMSE', math.sqrt(current_valid_sig_loss))
                print('valid_log RMSE', math.sqrt(current_valid_log_loss))
                print('exp accuracy', current_exp_accuracy.item())
                print('frac accuracy', current_frac_accuracy.item())
                print('in01 accuracy', current_in01_accuracy.item())
                print('in0100 accuracy', current_in0100_accuracy.item())

                total_acc += [current_exp_accuracy.item(),
                              current_frac_accuracy.item(),
                              current_in01_accuracy.item(),
                              current_in0100_accuracy.item()]
                total_rmse += [math.sqrt(current_valid_sig_loss),
                               math.sqrt(current_valid_log_loss)]

            if 'double' in args.evals:
                total_valid_add_sig_mse = []
                total_valid_add_log_mse = []
                total_valid_subtract_sig_mse = []
                total_valid_subtract_log_mse = []
                total_valid_cp_mse = []
                total_valid_cs_mse = []

                model.eval()
                add_correct_cnt = 0
                subtract_correct_cnt = 0
                for (numbers, results) in tqdm(double_valid_loader):
                    inputs = []
                    for number in numbers:
                        inputs.append(to_device(number, device))
                    with torch.no_grad():
                        output = model(task='double', input_dicts=inputs)

                        # add
                        add_sig_pred, add_exp_pred, add_log_pred = output[0], output[1], output[2]
                        add_val_gt, add_sig_gt, add_exp_gt, add_log_gt = results[0]
                        add_sig_gt = add_sig_gt.to(device)
                        add_exp_gt = add_exp_gt.to(device)
                        add_log_gt = add_log_gt.to(device)

                        # add, sig
                        add_sig_pred = add_sig_pred.view(add_sig_pred.size(0))
                        loss1 = MSE_loss_func(
                            add_sig_pred.view(add_sig_pred.size(0)),
                            add_sig_gt.view(add_sig_gt.size(0)))
                        total_valid_add_sig_mse.append(loss1.item())

                        # add, exp
                        add_exp_indexed = torch.argmax(add_exp_pred, dim=1)
                        add_exp_indexed = add_exp_indexed.view(-1, 1)
                        add_exp_gt = add_exp_gt.view(-1, 1)
                        add_correct_cnt += torch.sum(add_exp_indexed == add_exp_gt)

                        # add, log
                        add_log_pred = add_log_pred.view(add_log_pred.size(0))
                        loss2 = MSE_loss_func(add_log_pred.view(add_log_pred.size(0)),
                                              add_log_gt.view(add_log_gt.size(0)))
                        total_valid_add_log_mse.append(loss2.item())

                        # Subtract
                        subtract_sig_pred, subtract_exp_pred, subtract_log_pred, = output[3], output[4], output[5]
                        subtract_val_gt, subtract_sig_gt, subtract_exp_gt, subtract_log_gt = results[1]
                        subtract_sig_gt = subtract_sig_gt.to(device)
                        subtract_exp_gt = subtract_exp_gt.to(device)
                        subtract_log_gt = subtract_log_gt.to(device)

                        # subtract, sig
                        subtract_sig_pred = subtract_sig_pred.view(subtract_sig_pred.size(0))
                        loss3 = MSE_loss_func(subtract_sig_pred.view(subtract_sig_pred.size(0)),
                                              subtract_sig_gt.view(subtract_sig_gt.size(0)))
                        total_valid_subtract_sig_mse.append(loss3.item())

                        # subtract, exp
                        subtract_exp_indexed = torch.argmax(subtract_exp_pred, dim=1)
                        subtract_exp_indexed = subtract_exp_indexed.view(-1, 1)
                        subtract_exp_gt = subtract_exp_gt.view(-1, 1)
                        subtract_correct_cnt += torch.sum(subtract_exp_indexed == subtract_exp_gt)

                        # subtract, log
                        subtract_log_pred = subtract_log_pred.view(subtract_log_pred.size(0))
                        loss4 = MSE_loss_func(subtract_log_pred.view(subtract_log_pred.size(0)),
                                              subtract_log_gt.view(subtract_log_gt.size(0)))
                        total_valid_subtract_log_mse.append(loss4.item())

                        # CP, CS
                        cp_pred, cs_pred = output[6], output[7]
                        cp_gt, cs_gt = results[2]
                        cp_gt = cp_gt.to(device)
                        cs_gt = cs_gt.to(device)
                        loss5 = MSE_loss_func(cp_pred.view(cp_pred.size(0)), cp_gt.view(cp_gt.size(0)))
                        loss6 = MSE_loss_func(cs_pred.view(cs_pred.size(0)), cs_gt.view(cs_gt.size(0)))
                        total_valid_cp_mse.append(loss5.item())
                        total_valid_cs_mse.append(loss6.item())

                avg_valid_add_sig_mse = np.mean(total_valid_add_sig_mse)
                avg_valid_add_log_mse = np.mean(total_valid_add_log_mse)
                current_add_exp_accuracy = add_correct_cnt / len(double_valid_dataset)

                avg_valid_subtract_sig_mse = np.mean(total_valid_subtract_sig_mse)
                avg_valid_subtract_log_mse = np.mean(total_valid_subtract_log_mse)
                current_subtract_exp_accuracy = subtract_correct_cnt / len(double_valid_dataset)

                avg_valid_cp_mse = np.mean(total_valid_cp_mse)
                avg_valid_cs_mse = np.mean(total_valid_cs_mse)

                print('add sig RMSE', math.sqrt(avg_valid_add_sig_mse))
                print('add log RMSE', math.sqrt(avg_valid_add_log_mse))
                print('add exp accuracy', current_add_exp_accuracy.item())
                print('subtract sig RMSE', math.sqrt(avg_valid_subtract_sig_mse))
                print('subtract log RMSE', math.sqrt(avg_valid_subtract_log_mse))
                print('subtract exp accuracy', current_subtract_exp_accuracy.item())
                print('CP RMSE', avg_valid_cp_mse)
                print('CS RMSE', avg_valid_cs_mse)

                total_acc += [current_add_exp_accuracy.item(), current_subtract_exp_accuracy.item()]
                total_rmse += [math.sqrt(avg_valid_add_sig_mse),
                               math.sqrt(avg_valid_add_log_mse),
                               math.sqrt(avg_valid_subtract_sig_mse),
                               math.sqrt(avg_valid_subtract_log_mse),
                               avg_valid_cp_mse,
                               avg_valid_cs_mse]

            if 'multi' in args.evals:
                max_id_correct_cnt = 0
                for (numbers, results) in tqdm(multi_valid_loader):
                    inputs = []
                    for number in numbers:
                        inputs.append(to_device(number, device))

                    with torch.no_grad():
                        max_id_pred, feats = model('multi', inputs)
                        max_id_gt = results[0]
                        max_id_gt = max_id_gt.to(device)
                        max_id_indexed = torch.argmax(max_id_pred, dim=1)

                        max_id_correct_cnt += torch.sum(
                            max_id_indexed == max_id_gt)

                current_max_id_accuracy = max_id_correct_cnt / len(multi_valid_dataset)
                print('max_id acc', current_max_id_accuracy.item())

                total_acc += [current_max_id_accuracy.item()]

            print('\n\n')
            avg_acc = np.mean(total_acc)
            avg_rmse = np.mean(total_rmse)
            print('Epoch', epoch)
            print('avg acc:', avg_acc)
            print('avg rmse: ', avg_rmse)
            if avg_acc > best_acc[1]:
                best_acc = (epoch, avg_acc)
            if avg_rmse < best_rmse[1]:
                best_rmse = (epoch, avg_rmse)

            print('best acc', best_acc)
            print('best rmse', best_rmse)
            print('\n\n\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
