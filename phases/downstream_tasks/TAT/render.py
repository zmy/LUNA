import json


def func(id):
    for x in data:
        for que in x['questions']:
            if que['uid'] == id:
                print('----table----')
                for col in x['table']['table']:
                    print(col)
                print('----paragraph----')
                for para in x['paragraphs']:
                    print(para['text'])
                print('----question----')
                print(que['question'])
                print('----anwser----')
                print(que['answer'])
                return True
    return False


if __name__ == "__main__":
    with open('/storage/tuna-data/dataset_tagop/tatqa_dataset_dev.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    while True:
        line = input('please input id(enter q to quit): ')
        if line == 'q': break
        if not func(line): print('not find %s' % line)
