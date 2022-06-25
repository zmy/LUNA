# For other models, change the model_name to CharLSTM / RoBERTa / ValueEmbedding / Hybrid

python -m phases.single_number.eval.format \
--model_name RoBERTa \
--model_suffix='' \
--preprocess_type trivial \
--model_checkpoint_path='' \
--eval_num_class 3 \
--comment an_example \
--epoch_num 200 \
--seed 42 \
--emb_size 1024 \
--lstm_num_layers 3 \
--dataset_name unique \
--batch_size 16 \
--lr 1e-3 \
--device 0


# model_name: which model is used. 
#   Choose from: CharLSTM, RoBERTa, ValueEmbedding
#
# model_suffix: 
#   CharLSTM: Bi or None
#   RoBERTa: None
#   ValueEmbedding: _Sci or _Direct
#
# eval_num_class: Number in the eval task
#
# comment: A comment for evaluation result readability
#
# model_checkpoint_path:
#   Should be '' for RoBERTa and ValueEmbedding
#
# preprocess_type: ways to preprocess data
#   full, reverse, dec or trivial 
#   Should be the same as the one used during training
#
# epoch_num: Number of training epochs for the Evaluation model
#
# seed: random seed
#
# emb_size: Size of the embedding
#
# lstm_num_layers: Number of lstm layers for evaluating LSTMs
#
# dataset_name: Name of the dataset used 
#   unique 
# 
# batch_size:
#
# lr:
#
# device:
