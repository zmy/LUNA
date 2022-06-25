import json
import torch
from torch.utils.data import Dataset
from number_tokenizer.number_process import NUM_PROCESS_FUNCS
from number_tokenizer.numtok import NumTok

EXP_MAX, EXP_MIN = 7, -2

data_paths = {
    'sampled': (
        'exp/exp_data/double_train.json',
        'exp/exp_data/double_test.json'
    ),
}


class DoubleDataset(Dataset):
    def __init__(self, dataset_name, is_train, kept_keys, preprocess_type='trivial'):
        super(DoubleDataset, self).__init__()

        if is_train:
            with open(data_paths[dataset_name][0], 'r') as f:
                self.data_list = json.load(f)
            training_file_name = data_paths[dataset_name][0]
            print('training file: ', training_file_name)
        else:
            with open(data_paths[dataset_name][1], 'r') as f:
                self.data_list = json.load(f)
            validation_file_name = data_paths[dataset_name][1]
            print('validation file: ', validation_file_name)

        self.preprocess = lambda x: [NUM_PROCESS_FUNCS[preprocess_type](number) for number in x]
        self.kept_keys = kept_keys

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        return self.preprocess(self.data_list[index])

    def double_collate(self, data):
        numbers = []
        results = []
        for number_idx in range(len(data[0])):
            # In the double dataset, number_idx takes values 0 and 1
            batch_sig = [_[number_idx][1] for _ in data]
            batch_exp = [_[number_idx][2] for _ in data]
            batch_val = [_[number_idx][3] for _ in data]
            batch_number_string = [_[number_idx][4] for _ in data]

            batch_sig = torch.stack(batch_sig)
            batch_exp = torch.stack(batch_exp)
            batch_val = torch.stack(batch_val)
            batch_log = torch.log(torch.abs(batch_val) + 0.01)

            number_dict = {
                'batch_sig': batch_sig,
                'batch_exp': batch_exp,
                'batch_val': batch_val,
                'batch_log': batch_log,
            }
            token = NumTok.collate(batch_number_string, kept_keys=self.kept_keys, device='cpu')
            number_dict.update(token)
            numbers.append(number_dict)

        add_val_gt = torch.sum(torch.stack([_['batch_val'] for _ in numbers]), dim=0)
        sigexp = [list(NumTok.get_sci(str(num.item()))) for num in add_val_gt]
        for num in sigexp:
            num[1] = min(num[1], EXP_MAX)
            num[1] = max(num[1], EXP_MIN)
        add_sig_gt = torch.tensor([num[0] for num in sigexp])
        add_exp_gt = torch.tensor([num[1] for num in sigexp]) - EXP_MIN
        add_log_gt = torch.log(torch.abs(batch_val) + 0.01)
        results.append([add_val_gt, add_sig_gt, add_exp_gt, add_log_gt])

        subtract_val_gt = torch.sum(torch.stack((numbers[0]['batch_val'], -1*numbers[1]['batch_val'])), dim=0)
        sigexp = [list(NumTok.get_sci(str(num.item()))) for num in subtract_val_gt]
        for num in sigexp:
            num[1] = min(num[1], EXP_MAX)
            num[1] = max(num[1], EXP_MIN)
        subtract_sig_gt = torch.tensor([num[0] for num in sigexp])
        subtract_exp_gt = torch.tensor([num[1] for num in sigexp]) - EXP_MIN
        subtract_log_gt = torch.log(torch.abs(subtract_val_gt) + 0.01)
        results.append([subtract_val_gt, subtract_sig_gt, subtract_exp_gt, subtract_log_gt])

        cp_gt = []
        cs_gt = []
        for number_pair in data:
            num1 = number_pair[0][4]
            num2 = number_pair[1][4]
            cp_gt.append(get_cp(num1, num2))
            cs_gt.append(get_cs(num1, num2))
        cp_gt = torch.tensor(cp_gt, dtype=torch.float)
        cs_gt = torch.tensor(cs_gt, dtype=torch.float)
        results.append([cp_gt, cs_gt])

        return numbers, results


def get_cp(num1: str, num2: str):
    for char_idx, char in enumerate(num1):
        if char_idx >= len(num2):
            return len(num2)
        if char == num2[char_idx]:
            continue
        else:
            return char_idx
    return len(num1)


def get_cs(num1, num2):
    return get_cp(num1[::-1], num2[::-1])
