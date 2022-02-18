import argparse
import os
import pickle

from number_tokenizer import prepare_tokenizer
from .data.roberta_dataset import RobertaReader
from .data.tapas_dataset import TapasReader
from .options import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR


# python -m phases.downstream_tasks.TAT.prepare_dataset
def args_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--passage_length_limit", type=int, default=463)
    parser.add_argument("--question_length_limit", type=int, default=46)
    parser.add_argument("--redirect_huggingface_cache", type=int, default=1)
    return parser.parse_args()


def main(args):
    roberta_reader = RobertaReader(prepare_tokenizer('roberta', model_dir=args.model_dir,
                                                     redirect_huggingface_cache=args.redirect_huggingface_cache),
                                   args.passage_length_limit,
                                   args.question_length_limit)
    tapas_reader = TapasReader(prepare_tokenizer('tapas', model_dir=args.model_dir,
                                                 redirect_huggingface_cache=args.redirect_huggingface_cache),
                               args.passage_length_limit,
                               args.question_length_limit, sep_start="[CLS]", sep_end="[SEP]")
    for reader, model_name in zip([tapas_reader, roberta_reader], ['tapas', 'roberta']):
        for dm in ['train', 'dev']:
            in_file_path = os.path.join(args.input_dir, f"tatqa_dataset_{dm}.json")
            for use_numtok, name in zip([0, 1, 2], ['ori', 'num', 'both']):
                data = reader._read(in_file_path, use_numtok=use_numtok)
                out_file_path = os.path.join(args.output_dir, f"{model_name}_{name}_cache_{dm}.pkl")
                with open(out_file_path, "wb") as f:
                    pickle.dump(data, f)


if __name__ == '__main__':
    main(args_option())
