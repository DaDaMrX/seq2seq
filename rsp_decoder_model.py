import math
import types

import torch
from torch import nn
import transformers


def bert_for_masked_lm_forward(self, input_ids, kw_hidden, encoder_hidden, attention_mask=None, 
                               token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
    outputs = self.bert(input_ids,
                        kw_hidden=kw_hidden,
                        attention_mask=attention_mask,  # NOTE: add this line
                        encoder_hidden=encoder_hidden,  # NOTE: add this line
                        token_type_ids=token_type_ids,
                        position_ids=position_ids, 
                        head_mask=head_mask)

    sequence_output = outputs[0]
    prediction_scores = self.cls(sequence_output)

    outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
    if masked_lm_labels is not None:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        outputs = (masked_lm_loss,) + outputs

    return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


def bert_model_forward(self, input_ids, kw_hidden, encoder_hidden, attention_mask=None,
                       token_type_ids=None, position_ids=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * self.config.num_hidden_layers

    embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
    # kw_embedding_output = self.embeddings(kw_ids, position_ids=position_ids, token_type_ids=None)  # NOTE: add this line
    encoder_outputs = self.encoder(embedding_output,
                                    kw_hidden=kw_hidden,  # NOTE: add this line
                                    encoder_hidden=encoder_hidden,  # NOTE: add this line
                                    attention_mask=extended_attention_mask,
                                    head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


def bert_encoder_forward(self, hidden_states, kw_hidden, encoder_hidden, attention_mask=None, head_mask=None):
    all_hidden_states = ()
    all_attentions = ()
    for i, layer_module in enumerate(self.layer):
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # NOTE: add `encoder_hidden` to next line
        layer_outputs = layer_module(hidden_states, kw_hidden, encoder_hidden, attention_mask, head_mask[i])
        hidden_states = layer_outputs[0]
        if self.output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    # Add last layer
    if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if self.output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
        outputs = outputs + (all_attentions,)
    return outputs  # last-layer hidden state, (all hidden states), (all attentions)


def bert_layer_forward(self, hidden_states, kw_hidden, encoder_hidden, attention_mask=None, head_mask=None):
    attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
    attention_output = attention_outputs[0]

    # NOTE: add the 2 line blow
    attention_outputs = self.cross_atten(attention_output, encoder_hidden, attention_mask, head_mask)
    attention_output = attention_outputs[0]
    # NOTE: add the 2 line blow
    attention_outputs = self.kw_atten(attention_output, kw_hidden, attention_mask, head_mask)
    attention_output = attention_outputs[0]

    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
    return outputs


def cross_atten_block_forward(self, input_tensor, encoder_hidden, attention_mask=None, head_mask=None):
    self_outputs = self.self(input_tensor, encoder_hidden, attention_mask, head_mask)  # NOTE: add `encoder_hidden`
    attention_output = self.output(self_outputs[0], input_tensor)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


def cross_bert_self_atten_forward(self, hidden_states, encoder_hidden, attention_mask=None, head_mask=None):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(encoder_hidden)  # NOTE: change `hidden_states` to `encoder_hidden`
    mixed_value_layer = self.value(encoder_hidden)  # NOTE: change `hidden_states` to `encoder_hidden`

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # NOTE: Commented to remove attention mask (target to source)
    # if attention_mask is not None:
    #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    #     attention_scores = attention_scores + attention_mask

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

    outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
    return outputs


def kw_atten_block_forward(self, input_tensor, kw_hidden, attention_mask=None, head_mask=None):
    self_outputs = self.self(input_tensor, kw_hidden, attention_mask, head_mask)  # NOTE: add `kw_hidden`
    attention_output = self.output(self_outputs[0], input_tensor)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


def kw_bert_self_atten_forward(self, hidden_states, kw_hidden, attention_mask=None, head_mask=None):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(kw_hidden)  # NOTE: change `hidden_states` to `kw_hidden`
    mixed_value_layer = self.value(kw_hidden)  # NOTE: change `hidden_states` to `kw_hidden`

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # NOTE: Commented to remove attention mask (target to keywords)
    # if attention_mask is not None:
    #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    #     attention_scores = attention_scores + attention_mask

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

    outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
    return outputs


def mask_bert_self_atten_forward(self, hidden_states, attention_mask=None, head_mask=None):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # NOTE: Triangle mask (target to target)
    target_len = attention_scores.size(-1)
    mask = torch.tril(torch.ones(target_len, target_len))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    mask = mask.to(next(self.parameters()).device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    attention_scores = attention_scores + mask
    # NOTE: Commented
    # if attention_mask is not None:
    #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    #     attention_scores = attention_scores + attention_mask

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

    outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
    return outputs


class ResponseDecoder(torch.nn.Module):

    def __init__(self, pretrain_dir):
        super().__init__()
        config = transformers.BertConfig()
        config.num_hidden_layers = 6
        self.masked_lm = transformers.BertForMaskedLM.from_pretrained(pretrain_dir, config=config)
        for layer in self.masked_lm.bert.encoder.layer:
            layer.cross_atten = transformers.modeling_bert.BertAttention(self.masked_lm.config)
            layer.kw_atten = transformers.modeling_bert.BertAttention(self.masked_lm.config)

    def forward(self, input_ids, kw_hidden, encoder_hidden):
        self.bind_methods()
        return self.masked_lm(input_ids, kw_hidden, encoder_hidden)[0]

    def bind_methods(self):
        r"""Change forward method to add `encoder_hidden`.

        Architecture:
            (masked_lm): BertForMaskedLM                  [change forward]
                (bert): BertModel                         [change forward]
                    (embeddings): BertEmbeddings
                    (encoder): BertEncoder                [change forward] 
                        (layer): ModuleList
                            (0): BertLayer                    [change forward]
                                (attention): BertAttention        [change forward] [triangle mask]
                                    (self): BertSelfAttention
                                    (output): BertSelfOutput
                                (cross_atten): BertAttention      [change forward] [add model]
                                    (self): BertSelfAttention     [change forward]
                                    (output): BertSelfOutput
                                (kw_cross_atten): BertAttention      [change forward] [add model]
                                    (self): BertSelfAttention     [change forward]
                                    (output): BertSelfOutput
                                (intermediate): BertIntermediate
                                (output): BertSelfOutput
                    (pooler): BertPooler
                (cls): BertOnlyMLMHead
                    (predictions): BertLMPredictionHead
                    (transform): BertPredictionHeadTransform
                    (decoder): Linear
        """
        self.masked_lm.forward = types.MethodType(bert_for_masked_lm_forward, self.masked_lm)
        self.masked_lm.bert.forward = types.MethodType(bert_model_forward, self.masked_lm.bert)
        self.masked_lm.bert.encoder.forward = types.MethodType(bert_encoder_forward, self.masked_lm.bert.encoder)
        for layer in self.masked_lm.bert.encoder.layer:
            layer.forward = types.MethodType(bert_layer_forward, layer)
            layer.cross_atten.forward = types.MethodType(cross_atten_block_forward, layer.cross_atten)
            layer.cross_atten.self.forward = types.MethodType(cross_bert_self_atten_forward, layer.cross_atten.self)
            layer.kw_atten.forward = types.MethodType(kw_atten_block_forward, layer.kw_atten)
            layer.kw_atten.self.forward = types.MethodType(kw_bert_self_atten_forward, layer.kw_atten.self)
            layer.attention.self.forward = types.MethodType(mask_bert_self_atten_forward, layer.attention.self)
