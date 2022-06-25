from .config import NumBedConfig, MODEL_NAMES
from .numbed import NumBed
from .sembed import SemBed


def SelfAttention_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        number_prompt=None,
        number_prompt_mask=None,
):
    import torch
    import torch.nn as nn
    import math
    mixed_query_layer = self.query(hidden_states)

    # If this is instantiated as a cross-attention module, the keys
    # and values come from an encoder; the attention mask needs to be
    # such that the encoder's padding tokens are not attended to.
    is_cross_attention = encoder_hidden_states is not None

    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    elif number_prompt is not None:
        key_layer = self.transpose_for_scores(torch.cat((number_prompt[:,:,0],self.key(hidden_states)),dim=1))
        value_layer = self.transpose_for_scores(torch.cat((number_prompt[:,:,1],self.value(hidden_states)),dim=1))
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    query_layer = self.transpose_for_scores(mixed_query_layer)
    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in TapasModel forward() function)
        if number_prompt_mask is not None:
            number_prompt_mask=(1.0 - number_prompt_mask[:, None, None, :].to(attention_mask.dtype)) * -10000.0
            attention_mask=torch.cat((number_prompt_mask,attention_mask),dim=-1)
        attention_scores = attention_scores + attention_mask
    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs

# def SelfAttention_forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False,
# ):
#     import torch
#     import torch.nn as nn
#     import math
#     mixed_query_layer = self.query(hidden_states)
#
#     # If this is instantiated as a cross-attention module, the keys
#     # and values come from an encoder; the attention mask needs to be
#     # such that the encoder's padding tokens are not attended to.
#     is_cross_attention = encoder_hidden_states is not None
#
#     if is_cross_attention and past_key_value is not None:
#         # reuse k,v, cross_attentions
#         key_layer = past_key_value[0]
#         value_layer = past_key_value[1]
#         attention_mask = encoder_attention_mask
#     elif is_cross_attention:
#         key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
#         value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
#         attention_mask = encoder_attention_mask
#     elif past_key_value is not None:
#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))
#         key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
#         value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
#     elif getattr(self,'number_prompt',None) is not None:
#         key_layer = self.transpose_for_scores(torch.cat((self.number_prompt[:,:,0],self.key(hidden_states)),dim=1))
#         value_layer = self.transpose_for_scores(torch.cat((self.number_prompt[:,:,1],self.value(hidden_states)),dim=1))
#     else:
#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))
#
#     query_layer = self.transpose_for_scores(mixed_query_layer)
#
#     if self.is_decoder:
#         past_key_value = (key_layer, value_layer)
#
#     # Take the dot product between "query" and "key" to get the raw attention scores.
#     attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#     attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#     if attention_mask is not None:
#         # Apply the attention mask is (precomputed for all layers in TapasModel forward() function)
#         if getattr(self,'number_prompt_mask',None) is not None:
#             number_prompt_mask=(1.0 - self.number_prompt_mask[:, None, None, :].to(attention_mask.dtype)) * -10000.0
#             attention_mask=torch.cat((number_prompt_mask,attention_mask),dim=-1)
#         attention_scores = attention_scores + attention_mask
#
#     # Normalize the attention scores to probabilities.
#     attention_probs = nn.Softmax(dim=-1)(attention_scores)
#
#     # This is actually dropping out entire tokens to attend to, which might
#     # seem a bit unusual, but is taken from the original Transformer paper.
#     attention_probs = self.dropout(attention_probs)
#
#     # Mask heads if we want to
#     if head_mask is not None:
#         attention_probs = attention_probs * head_mask
#
#     context_layer = torch.matmul(attention_probs, value_layer)
#
#     context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#     new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#     context_layer = context_layer.view(*new_context_layer_shape)
#
#     outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#     if self.is_decoder:
#         outputs = outputs + (past_key_value,)
#     return outputs