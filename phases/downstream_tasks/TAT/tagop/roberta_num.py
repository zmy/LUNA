from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
from transformers import RobertaModel

from number_encoder import NumBed, NumBedConfig
from .util import FFNLayer
from .util import get_numbers_from_reduce_sequence, get_span_tokens_from_paragraph, \
    get_span_tokens_from_table, get_single_span_tokens_from_table, get_single_span_tokens_from_paragraph
from ..data.data_util import SCALE, OPERATOR_CLASSES_, id2OPERATOR_CLASSES_
from ..data.tatqa_metric import TaTQAEmAndF1
import os


class RobertaNum(nn.Module):
    def __init__(self,
                 tokenizer,
                 hidden_size: int,
                 dropout_prob: float,
                 use_newtag,
                 model_name,
                 checkpoint_path,
                 model_dir,
                 redirect_huggingface_cache
                 ):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(os.path.join(model_dir, "roberta.large"))
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.use_numbed = model_name != 'zero'
        if self.use_numbed:
            number_model_config = NumBedConfig(model_name=model_name,
                                               encoder_name='RoBERTa',
                                               checkpoint_path=checkpoint_path)
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

    def _encode(self, input_ids: torch.LongTensor, numtok_dict: Dict, attention_mask: torch.LongTensor,
                token_type_ids: torch.LongTensor) -> torch.Tensor:
        if self.use_numbed:
            num_embed = self.numbed.embedding(input_ids, numtok_dict, self.tokenizer.num_token_id)
        else:
            num_embed = None
        txt_embed = self.encoder.embeddings.word_embeddings(input_ids)
        if num_embed is None:
            input_embedding = txt_embed
        else:
            input_embedding = txt_embed + num_embed
        return self.encoder(
            inputs_embeds=input_embedding,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def develop(self, output_dict, tag_prediction, token_type_ids, input_index, \
                operator_prediction, scale_prediction, batch_size, paragraph_tokens, \
                table_cell_tokens, paragraph_numbers, table_cell_numbers, table_sequence_reduce_mean_output, \
                paragraph_sequence_reduce_mean_output, question_ids, gold_answers, operator_labels):
        output_dict["answer"] = []
        output_dict["scale"] = []

        table_tag_reduce_max_prediction, _ = scatter_max(
            src=tag_prediction[..., 1] * (token_type_ids == 1) - 1e10 * (token_type_ids != 1),
            index=input_index * (token_type_ids == 1),
            dim=1,
        )
        table_reduce_mask = (_ != token_type_ids.size(1)) & (_ != -1)
        table_reduce_mask[:, 0] = False

        paragraph_tag_reduce_max_prediction, _ = scatter_max(
            src=tag_prediction[..., 1] * (token_type_ids == 2) - 1e10 * (token_type_ids != 2),
            index=input_index * (token_type_ids == 2),
            dim=1,
        )
        paragraph_reduce_mask = (_ != token_type_ids.size(1)) & (_ != -1)
        paragraph_reduce_mask[:, 0] = False

        masked_table_tag_reduce_max_prediction = table_tag_reduce_max_prediction.masked_fill(~table_reduce_mask,
                                                                                             -1e10)
        masked_paragraph_tag_reduce_max_prediction = paragraph_tag_reduce_max_prediction.masked_fill(
            ~paragraph_reduce_mask, -1e10)

        table_cell_tag_prediction_score = masked_table_tag_reduce_max_prediction.cpu().numpy()
        table_cell_tag_prediction = (masked_table_tag_reduce_max_prediction > -0.6931).long().cpu().numpy()
        paragraph_token_tag_prediction_score = masked_paragraph_tag_reduce_max_prediction.cpu().numpy()
        paragraph_token_tag_prediction = (masked_paragraph_tag_reduce_max_prediction > -0.6931).long().cpu().numpy()

        predicted_operator_class = torch.argmax(operator_prediction, dim=-1).cpu().numpy()
        predicted_scale_class = torch.argmax(scale_prediction, dim=-1).cpu().numpy()
        #################
        for bsz in range(batch_size):
            # test_reduce_mean(paragraph_tag_prediction[bsz], paragraph_index[bsz])
            if predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TEXT"]:
                paragraph_selected_span_tokens = get_single_span_tokens_from_paragraph(
                    paragraph_token_tag_prediction[bsz],
                    paragraph_token_tag_prediction_score[bsz],
                    paragraph_tokens[bsz]
                )
                answer = paragraph_selected_span_tokens
                answer = sorted(answer)
            elif predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TABLE"]:
                table_selected_tokens = get_single_span_tokens_from_table(
                    table_cell_tag_prediction[bsz],
                    table_cell_tag_prediction_score[bsz],
                    table_cell_tokens[bsz])
                answer = table_selected_tokens
                answer = sorted(answer)
            elif predicted_operator_class[bsz] == self.OPERATOR_CLASSES["MULTI_SPAN"]:
                paragraph_selected_span_tokens = \
                    get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz])
                table_selected_tokens = \
                    get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz])
                answer = paragraph_selected_span_tokens + table_selected_tokens
                answer = sorted(answer)
            elif predicted_operator_class[bsz] == self.OPERATOR_CLASSES["COUNT"]:
                paragraph_selected_tokens = \
                    get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz])
                table_selected_tokens = \
                    get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz])
                answer = len(paragraph_selected_tokens) + len(table_selected_tokens)
            else:
                if predicted_operator_class[bsz] in {self.OPERATOR_CLASSES["SUM"], self.OPERATOR_CLASSES["TIMES"],
                                                     self.OPERATOR_CLASSES["AVERAGE"]}:
                    paragraph_selected_numbers = \
                        get_numbers_from_reduce_sequence(paragraph_token_tag_prediction[bsz],
                                                         paragraph_numbers[bsz])
                    table_selected_numbers = \
                        get_numbers_from_reduce_sequence(table_cell_tag_prediction[bsz], table_cell_numbers[bsz])
                    selected_numbers = paragraph_selected_numbers + table_selected_numbers
                    if not selected_numbers:
                        answer = ""
                    elif predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SUM"]:
                        answer = np.around(np.sum(selected_numbers), 4)
                    elif predicted_operator_class[bsz] == self.OPERATOR_CLASSES["TIMES"]:
                        answer = np.around(np.prod(selected_numbers), 4)
                    elif predicted_operator_class[bsz] == self.OPERATOR_CLASSES["AVERAGE"]:
                        answer = np.around(np.mean(selected_numbers), 4)
                else:
                    masked_tag_reduce_max_prediction = torch.cat(
                        (masked_table_tag_reduce_max_prediction[bsz], masked_paragraph_tag_reduce_max_prediction[bsz]))
                    sort_value, sorted_index = torch.sort(masked_tag_reduce_max_prediction, descending=True)
                    if sort_value.size(0) < 2 or sort_value[1] < -1e5:
                        answer = ""
                    else:
                        sorted_index = sorted_index[:2]
                        sorted_index = torch.sort(sorted_index)[0].cpu().numpy()
                        operand = []
                        vector = []
                        for _sorted_index in sorted_index:
                            if _sorted_index < masked_table_tag_reduce_max_prediction.size(1):
                                operand.append(table_cell_numbers[bsz][_sorted_index - 1])
                                vector.append(table_sequence_reduce_mean_output[bsz, _sorted_index - 1])
                            else:
                                operand.append(paragraph_numbers[bsz][
                                                   _sorted_index - masked_table_tag_reduce_max_prediction.size(1) - 1])
                                vector.append(paragraph_sequence_reduce_mean_output[
                                                  bsz, _sorted_index - masked_table_tag_reduce_max_prediction.size(
                                                      1) - 1])
                        operand_one, operand_two = operand
                        predicted_order = torch.argmax(self.order_predictor(torch.cat(vector))).item()
                        if predicted_order == 1:
                            operand_one, operand_two = operand_two, operand_one
                        if np.isnan(operand_one) or np.isnan(operand_two):
                            answer = ""
                        else:
                            if predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIFF"]:
                                answer = np.around(operand_one - operand_two, 4)
                            elif predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIVIDE"]:
                                if operand_two == 0:
                                    answer = 1.0
                                else:
                                    answer = np.around(operand_one / operand_two, 4)
                                if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                    answer = answer * 100
                            elif predicted_operator_class[bsz] == self.OPERATOR_CLASSES["CHANGE_RATIO"]:
                                if operand_two == 0:
                                    answer = 0.0
                                else:
                                    answer = np.around(operand_one / operand_two - 1, 4)
                                if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                    answer = answer * 100

            output_dict["answer"].append(answer)
            output_dict["scale"].append(SCALE[int(predicted_scale_class[bsz])])
            question_id = None if question_ids is None else question_ids[bsz]
            self._metrics(gold_answers[bsz], id2OPERATOR_CLASSES_[operator_labels[bsz].item()], \
                          answer, id2OPERATOR_CLASSES_[predicted_operator_class[bsz]], \
                          SCALE[int(predicted_scale_class[bsz])], question_id)

    def forward(self,
                input_ids: torch.LongTensor,
                numtok_dict: Dict,
                attention_mask: torch.LongTensor,
                token_type_ids: torch.LongTensor,
                input_index: torch.LongTensor,
                tag_labels: torch.LongTensor,
                operator_labels: torch.LongTensor,
                scale_labels: torch.LongTensor,
                number_order_labels: torch.LongTensor,
                gold_answers: str,
                paragraph_tokens: List[List[str]],
                table_cell_numbers: List[np.ndarray],
                paragraph_numbers: List[np.ndarray],
                table_cell_tokens: List[List[str]],
                token_type_ids_for_encoder: torch.LongTensor = None,
                dev=False,
                question_ids=None,
                **kwargs
                ):
        device = input_ids.device
        if token_type_ids_for_encoder is None:
            token_type_ids_for_encoder = torch.zeros_like(input_ids)
        outputs = self._encode(input_ids, numtok_dict, attention_mask, token_type_ids_for_encoder)
        sequence_output = outputs[0]
        batch_size = sequence_output.shape[0]
        cls_output = sequence_output[:, 0, :]
        operator_prediction = self.operator_predictor(cls_output)
        tag_prediction = self.tag_predictor(sequence_output)
        tag_prediction = F.log_softmax(tag_prediction, dim=-1)
        table_reduce_mean = (sequence_output * (token_type_ids == 1).unsqueeze(-1)).sum(1) / (
                (token_type_ids == 1).unsqueeze(-1).sum(1) + 1e-7)
        paragraph_reduce_mean = (sequence_output * (token_type_ids == 2).unsqueeze(-1)).sum(1) / (
                (token_type_ids == 2).unsqueeze(-1).sum(1) + 1e-7)
        cat_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)
        scale_prediction = self.scale_predictor(cat_output)
        table_tag_reduce_max_label, _ = scatter_max(
            src=tag_labels * (token_type_ids == 1),
            index=input_index * (token_type_ids == 1),
            dim=1,
        )
        table_reduce_mask = (_ != token_type_ids.size(1)) & (_ != -1)
        table_reduce_mask[:, 0] = False
        table_sequence_reduce_mean_output = scatter_mean(
            src=sequence_output,
            index=(input_index * (token_type_ids == 1)).unsqueeze(-1),
            dim=1,
        )
        paragraph_tag_reduce_max_label, _ = scatter_max(
            src=tag_labels * (token_type_ids == 2),
            index=input_index * (token_type_ids == 2),
            dim=1,
        )
        paragraph_reduce_mask = (_ != token_type_ids.size(1)) & (_ != -1)
        paragraph_reduce_mask[:, 0] = False
        paragraph_sequence_reduce_mean_output = scatter_mean(
            src=sequence_output,
            index=(input_index * (token_type_ids == 2)).unsqueeze(-1),
            dim=1,
        )
        top_2_order_ground_truth = torch.zeros_like(number_order_labels)
        top_2_sequence_output_bw = torch.zeros(batch_size, 2 * sequence_output.shape[2]).to(device)
        ground_truth_index = 0
        for bsz in range(batch_size):
            if operator_labels[bsz].item() in {self.OPERATOR_CLASSES["DIVIDE"], self.OPERATOR_CLASSES["DIFF"],
                                               self.OPERATOR_CLASSES["CHANGE_RATIO"]}:
                cat_feature = torch.cat((table_sequence_reduce_mean_output[bsz][table_tag_reduce_max_label[bsz] == 1], \
                                         paragraph_sequence_reduce_mean_output[bsz][
                                             paragraph_tag_reduce_max_label[bsz] == 1]), axis=0)  # N,dim
                if cat_feature.size(0) < 2: continue
                top_2_sequence_output_bw[ground_truth_index] = cat_feature[:2].view(-1)
                top_2_order_ground_truth[ground_truth_index] = number_order_labels[bsz]
                ground_truth_index += 1
        top_2_order_prediction_bw = self.order_predictor(top_2_sequence_output_bw[:ground_truth_index])
        top_2_order_ground_truth = top_2_order_ground_truth[:ground_truth_index]
        # down, calculate the loss
        output_dict = {}
        operator_prediction_loss = self.criterion(operator_prediction, operator_labels)
        scale_prediction_loss = self.criterion(scale_prediction, scale_labels)
        tag_prediction_loss = self.NLLLoss(tag_prediction[token_type_ids != 0], tag_labels[token_type_ids != 0])
        if ground_truth_index != 0:
            top_2_order_prediction_bw = F.log_softmax(top_2_order_prediction_bw, dim=-1)
            top_2_order_prediction_loss = self.NLLLoss(top_2_order_prediction_bw, top_2_order_ground_truth)
        else:
            top_2_order_prediction_loss = torch.tensor(0, dtype=torch.float).to(device)

        output_dict["loss"] = [operator_prediction_loss, scale_prediction_loss, tag_prediction_loss,
                               top_2_order_prediction_loss]
        if dev:
            self.develop(output_dict, tag_prediction, token_type_ids, input_index, \
                         operator_prediction, scale_prediction, batch_size, paragraph_tokens, \
                         table_cell_tokens, paragraph_numbers, table_cell_numbers, table_sequence_reduce_mean_output, \
                         paragraph_sequence_reduce_mean_output, question_ids, gold_answers, operator_labels)
        return output_dict
