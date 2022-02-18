import os
from tqdm import tqdm
import json
from collections import defaultdict
input_dir=r"C:\Users\v-hongweihan\Downloads\NumberDataset"
with open(os.path.join(input_dir,'statistics.json'),encoding='utf8')as f:
    statistics=json.load(f)

with open(os.path.join(input_dir,'text.json.num'),encoding='utf8')as f:
    number=json.load(f)

count=defaultdict(dict)
for dataset in statistics:
    os.makedirs(os.path.join(input_dir, 'numbers', dataset),exist_ok=True)
    for part in statistics[dataset]:
        output=[]
        for id in tqdm(statistics[dataset][part]):
            id=str(id)
            if id in number:
                for num in number[id]:
                    output.append(num['Resolution']['value'])
        with open(os.path.join(input_dir, 'numbers', dataset,'%s.json'%part),'w',encoding='utf8')as f:
            json.dump(output,f,indent=4)
        count[dataset][part]=len(output)

with open(os.path.join(input_dir, 'numbers', 'count.json'),'w',encoding='utf8')as f:
    json.dump(count,f,indent=4)


