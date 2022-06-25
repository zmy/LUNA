import json
import torch
from torch.utils.data import Dataset
from number_tokenizer.number_process import NUM_PROCESS_FUNCS
from number_tokenizer.numtok import NumTok

data_paths = {
    'sampled': (
        'exp/exp_data/multi_train.json',
        'exp/exp_data/multi_test.json'
    ),
}


class MultiDataset(Dataset):
    def __init__(self, dataset_name, is_train, kept_keys, preprocess_type='trivial'):
        super(MultiDataset, self).__init__()

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

    def multi_collate(self, data):
        numbers = []
        results = []
        for number_idx in range(len(data[0])):
            # In the multi dataset, number_idx takes values 0 to 4
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
                'batch_log': batch_log
            }
            token = NumTok.collate(batch_number_string, kept_keys=self.kept_keys, device='cpu')
            number_dict.update(token)
            numbers.append(number_dict)

        max_id_gt = torch.argmax(torch.stack([_['batch_val'] for _ in numbers]), dim=0)
        results = [max_id_gt]

        return numbers, results
