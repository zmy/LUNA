import os

os.sys.path.insert(0, '')
import json
from phases.numerical_field.dataset.utils import parse_string
import numpy as np
from collections import Counter
from tqdm import tqdm


def get_cdf(values):
    total = len(values)
    values = Counter(values)
    values = sorted(values.items(), key=lambda x: x[0])
    accumulate = 0
    cdf = []
    for k, v in values:
        accumulate += v
        cdf.append((k, accumulate / total))
    return cdf


if __name__ == '__main__':
    with open('data/PretrainDataset/all_table.json', encoding='utf8') as f:
        tables = json.load(f)

    single_column_ratios = []
    column_ratios = []
    single_row_ratios = []
    row_ratios = []

    for table in tqdm(tables.values()):
        num_table = []
        for row in table:
            num_row = []
            for cell in row:
                _, numbers = parse_string(cell)
                num_row.append(len(numbers))
            num_table.append(num_row)

        if len(num_table) == 0: continue
        num_table = np.array(num_table)
        if num_table.shape[1] == 0: continue

        single_column_ratios.extend(((num_table == 1).sum(0) / num_table.shape[0]).tolist())
        single_row_ratios.extend(((num_table == 1).sum(1) / num_table.shape[1]).tolist())
        column_ratios.extend(((num_table > 0).sum(0) / num_table.shape[0]).tolist())
        row_ratios.extend(((num_table > 0).sum(1) / num_table.shape[1]).tolist())

    for name in tqdm(["single_column_ratios", "single_row_ratios", "column_ratios", "row_ratios"]):
        cdf = get_cdf(eval(name))
        with open('data/analysis/%s_cdf.json' % name, 'w') as f:
            json.dump(cdf, f)
