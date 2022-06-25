import json
import torch
from torch.utils.data import Dataset
from number_tokenizer.number_process import NUM_PROCESS_FUNCS
from number_tokenizer.vocab import UNTIDY
from number_tokenizer import NumTok

FRAC_MAX = 3

data_paths = {
    'sampled': (
        'exp/exp_data/single_train.json',
        'exp/exp_data/single_test.json'
    ),
}


class SingleDataset(Dataset):
    def __init__(self, dataset_name, is_train, kept_keys, preprocess_type='trivial'):
        super(SingleDataset, self).__init__()

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
        self.preprocess = NUM_PROCESS_FUNCS[preprocess_type]
        self.kept_keys = kept_keys

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        return self.preprocess(self.data_list[index])

    def single_collate(self, data):

        # Prepare labels
        batch_sig = [_[1] for _ in data]
        batch_exp = [_[2] for _ in data]
        batch_val = [_[3] for _ in data]
        batch_number_string = [_[4] for _ in data]
        batch_sig = torch.stack(batch_sig)
        batch_exp = torch.stack(batch_exp)
        batch_val = torch.stack(batch_val)
        batch_log = torch.log(torch.abs(batch_val) + 0.01)
        batch_in01 = torch.logical_and(batch_val < 1, batch_val > 0).long()
        batch_in0100 = torch.logical_and(batch_val < 100, batch_val > 0).long()
        batch_frac = []
        for string in batch_number_string:
            if '.' not in string:
                batch_frac.append(0)
                continue
            string = ''.join([char for char in string if char not in UNTIDY])
            frac = len(string.split('.')[1])
            frac = min(frac, FRAC_MAX)
            batch_frac.append(frac)
        batch_frac = torch.tensor(batch_frac).long()

        # Prepare inputs
        token = NumTok.collate(batch_number_string, kept_keys=self.kept_keys, device='cpu')
        number_dict = {
            'batch_sig': batch_sig,
            'batch_exp': batch_exp,
            'batch_val': batch_val,
            'batch_log': batch_log,
            'batch_frac': batch_frac,
            'batch_in01': batch_in01,
            'batch_in0100': batch_in0100,
        }
        token.update(number_dict)
        return token


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    single_train_dataset = SingleDataset('sampled', is_train=True,
                                         preprocess_type='trivial')
    single_train_loader = DataLoader(single_train_dataset,
                                     batch_size=32,
                                     num_workers=16,
                                     pin_memory=True,
                                     shuffle=True,
                                     collate_fn=single_collate)

    for idx, batch in enumerate(single_train_loader):
        if idx > 1:
            break
        print(batch['input_ids'])
        print(batch['attention_mask'])
