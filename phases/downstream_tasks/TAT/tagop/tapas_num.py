import torch.nn as nn
from transformers import TapasModel

from number_encoder import NumBed, NumBedConfig,SelfAttention_forward
from .roberta_num import RobertaNum
from .util import FFNLayer
from ..data.data_util import SCALE, OPERATOR_CLASSES_
from ..data.tatqa_metric import TaTQAEmAndF1
import os


class TapasNum(RobertaNum):
    def __init__(self,
                 tokenizer,
                 hidden_size: int,
                 dropout_prob: float,
                 use_newtag,
                 model_name,
                 checkpoint_path,
                 model_dir,
                 redirect_huggingface_cache,
                 use_prompt
                 ):
        nn.Module.__init__(self)
        self.encoder = TapasModel.from_pretrained('google/tapas-base-masklm',
                                                  cache_dir=os.path.join(model_dir,
                                                                         'huggingface') if redirect_huggingface_cache else None)
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.use_numbed = model_name != 'zero'
        if self.use_numbed:
            number_model_config = NumBedConfig(model_name=model_name,
                                               encoder_name='TaPas',
                                               checkpoint_path=checkpoint_path,
                                               prompt_layers=None if not use_prompt else 12)
            self.numbed = NumBed(number_model_config)
        # operator predictor
        self.operator_predictor = FFNLayer(hidden_size, hidden_size, len(OPERATOR_CLASSES_), dropout_prob)
        # scale predictor
        self.scale_predictor = FFNLayer(3 * hidden_size, hidden_size, len(SCALE), dropout_prob)
        # tag predictor: two-class classification
        self.tag_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
        self.order_predictor = FFNLayer(2 * hidden_size, hidden_size, 2, dropout_prob)
        # criterion for operator/scale loss calculation
        self.criterion = nn.CrossEntropyLoss()
        # NLLLoss for tag_prediction
        self.NLLLoss = nn.NLLLoss(reduction="sum")
        self.OPERATOR_CLASSES = OPERATOR_CLASSES_
        self._metrics = TaTQAEmAndF1()
        self.tokenizer = tokenizer
        self.use_newtag = use_newtag
        self.use_prompt = use_prompt
