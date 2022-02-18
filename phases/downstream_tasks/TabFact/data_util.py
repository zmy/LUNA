import pandas as pd
import torch

from number_tokenizer.numtok import NumTok


def prepare_data(data_itr, is_train, args):
    data = data_itr

    data_sample = torch.utils.data.distributed.DistributedSampler(data, num_replicas=args.nprocs, rank=args.rank,
                                                                  seed=args.seed)
    batch_size = args.batch_size
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=int(
        batch_size // args.nprocs // args.gradient_accumulation_steps), shuffle=False,
                                              pin_memory=True,
                                              sampler=data_sample,
                                              drop_last=is_train)
    return data, data_loader, data_sample


def read_text_as_pandas_table(table_text: str):
    table = pd.DataFrame([x.split('#') for x in table_text.split('\n')[1:-1]],
                         columns=[x for x in table_text.split('\n')[0].split('#')]).fillna('')
    table = table.astype(str)
    return table


def roberta_string_tokenize(string, tokenizer, use_numtok):
    if not string:
        return [], []
    if use_numtok == 1:
        new_text, number_triplets = NumTok.replace_numbers(string, do_lower_case=False)  # NEWLY ADDED SECTION
        number_strings = [t[0] for t in number_triplets]
    elif use_numtok == 2:
        new_text, number_triplets = NumTok.replace_numbers(string, do_lower_case=False, keep_origin=1)  # NEWLY ADDED SECTION
        number_strings = [t[0] for t in number_triplets]
    else:
        new_text, number_strings = string, []

    tokens = tokenizer.tokenize(new_text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids, number_strings


def roberta_table_tokenize(table_text, tokenizer, use_numtok):
    table = [x.split('#') for x in table_text.split('\n')[:-1]]
    table_ids = []
    table_number_strings = []
    if len(table) > 0:
        for j in range(len(table[0])):
            for i in range(len(table)):
                cell_ids, number_strings = roberta_string_tokenize(table[i][j], tokenizer, use_numtok=use_numtok)
                if not cell_ids:
                    continue
                table_ids += cell_ids
                table_number_strings += number_strings  # NEWLY ADDED

    return table_ids, table_number_strings
