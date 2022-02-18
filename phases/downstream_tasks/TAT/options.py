from argparse import ArgumentParser

TABLE_ENCODERS = ['roberta', 'tapas', 'bert']
MODES = ['train', 'dev']
DEFAULT_DATA_DIR = "/storage/tuna-data/dataset_tagop"
DEFAULT_MODEL_DIR = "/storage/tuna-models"


def add_train_args(parser: ArgumentParser):
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_epoch", default=25, type=int, help="max epoch.")
    parser.add_argument("--log_per_updates", default=20, type=int, help="log_per_updates.")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="weight decay.")
    parser.add_argument("--learning_rate", default=1.5e-4, type=float, help="learning rate.")
    parser.add_argument("--grad_clipping", default=1.0, type=float, help="gradient clip.")
    parser.add_argument('--warmup', type=float, default=0.06,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_schedule", default="warmup_linear", type=str, help="warmup schedule.")
    parser.add_argument("--optimizer", default="adam", type=str, help="train optimizer.")
    parser.add_argument('--seed', type=int, default=2018, help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument('--batch_size_per_node', type=int, default=2, help="batch size.")
    parser.add_argument('--eval_batch_size_per_node', type=int, default=2, help="eval batch size.")
    parser.add_argument("--eps", default=1e-6, type=float, help="ema gamma.")


def add_bert_args(parser: ArgumentParser):
    parser.add_argument("--bert_learning_rate", type=float, default=5e-6, help="bert learning rate.")
    parser.add_argument("--bert_weight_decay", type=float, default=0.01, help="bert weight decay.")
    parser.add_argument("--model_dir", type=str, help="model folder.", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--redirect_huggingface_cache", type=int, default=1)
