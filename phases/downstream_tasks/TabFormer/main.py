import sys

import setproctitle
import transformers

setproctitle.setproctitle("Tabformer")
import logging
from os import makedirs
from os.path import join

from number_encoder import NumBedConfig
from transformers import Trainer, TrainingArguments

from .args import define_main_parser
from .dataset.card import TransactionDataset
from .misc.utils import random_split_dataset
from .models.modules import TabFormerBertLM

from .dataset.datacollator import TransDataCollatorForLanguageModeling, numtok_collate_fn


class TabFormerTrainer(Trainer):
    """Customized trainer for training TabFormer w/ and w/o number net."""

    def __init__(self,
                 use_numtok,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_numtok = use_numtok

    def training_step(self, model, inputs):
        if self.use_numtok:
            outputs = model(input_ids=inputs['input_ids'], tokenized_number=inputs['tokenized_numbers'], mlm=True)
            loss = outputs[0]

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            # TODO: do_grad_scaling, deepspeed are newly supported by transformers, low version package may not support
            # this. (The following code is tested on transformers 4.15.0. If you need to use these features, please
            # upgrade the transformers package and uncomment the if conditions.)
            # if self.do_grad_scaling:
            #     self.scaler.scale(loss).backward()
            # elif self.deepspeed:
            #     loss = self.deepspeed.backward(loss)
            # else:
            #     loss.backward()

            loss.backward()

            return loss.detach()

        else:
            return super().training_step(model, inputs)


def main(args):
    # Sets up training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        save_strategy='epoch',
        save_total_limit=1,
        # wandb arguments.
        report_to='wandb',
    )

    # Setup logging for preparing data.
    logger = logging.getLogger(__name__)
    log = logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Builds dataset.
    if args.data_type == 'card':
        dataset = TransactionDataset(root=args.data_root,
                                     fname=args.data_fname,
                                     fextension=args.data_extension,
                                     vocab_dir=args.output_dir,
                                     nrows=args.nrows,
                                     user_ids=args.user_ids,
                                     mlm=args.mlm,
                                     cached=args.cached,
                                     stride=args.stride,
                                     flatten=args.flatten,
                                     return_labels=False,
                                     skip_user=args.skip_user,
                                     use_numtok=args.use_numtok,
                                     log=logger)
    else:
        raise Exception(f"data type '{args.data_type}' not defined")

    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    # split dataset into train, val, test [0.6. 0.2, 0.2]
    totalN = len(dataset)
    trainN = int(0.6 * totalN)

    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN

    assert totalN == trainN + valN + testN

    lengths = [trainN, valN, testN]

    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

    if args.use_numtok:
        number_model_config = NumBedConfig(model_name=args.number_model_config)
    else:
        number_model_config = None

    # Builds model.
    if args.lm_type == "bert":
        tab_net = TabFormerBertLM(custom_special_tokens,
                                  vocab=vocab,
                                  field_ce=args.field_ce,
                                  flatten=args.flatten,
                                  ncols=dataset.ncols,
                                  field_hidden_size=args.field_hs,
                                  use_numtok=args.use_numtok,
                                  number_model_config=number_model_config,
                                  use_replace=args.use_replace,
                                  use_reg_loss=args.use_reg_loss,
                                  data_type=args.data_type
                                  )
    else:
        raise NotImplementedError(f'{args.lm_type} not supported!')

    log.info(f"model initiated: {tab_net.model.__class__}")

    if args.use_numtok:
        data_collator = numtok_collate_fn
    else:
        data_collator = TransDataCollatorForLanguageModeling(tokenizer=tab_net.tokenizer, mlm=args.mlm,
                                                             mlm_probability=args.mlm_prob)

    # Initializes the customized Trainer.
    trainer = TabFormerTrainer(
        use_numtok=args.use_numtok,
        model=tab_net.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Starts training.
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":

    parser = define_main_parser()
    opts = parser.parse_args()

    opts.log_dir = join(opts.output_dir, "logs")
    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)

    if not opts.mlm and opts.lm_type == "bert":
        raise Exception("Error: Bert needs '--mlm' option. Please re-run with this flag included.")

    main(opts)
