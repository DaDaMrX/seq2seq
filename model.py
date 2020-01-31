import math
import random
import types

import torch
from torch import nn
import transformers

from rsp_decoder_model import ResponseDecoder


def bert_for_masked_lm_forward(self, input_ids, encoder_hidden, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            masked_lm_labels=None):
    outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
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


def bert_model_forward(self, input_ids, encoder_hidden, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
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
    encoder_outputs = self.encoder(embedding_output,
                                    encoder_hidden,  # NOTE: add this line
                                    extended_attention_mask,
                                    head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


def bert_encoder_forward(self, hidden_states, encoder_hidden, attention_mask=None, head_mask=None):
    all_hidden_states = ()
    all_attentions = ()
    for i, layer_module in enumerate(self.layer):
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # NOTE: add `encoder_hidden` to next line
        layer_outputs = layer_module(hidden_states, encoder_hidden, attention_mask, head_mask[i])
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


def bert_layer_forward(self, hidden_states, encoder_hidden, attention_mask=None, head_mask=None):
    attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
    attention_output = attention_outputs[0]
    # NOTE: add the 2 line blow
    attention_outputs = self.cross_atten(attention_output, encoder_hidden, attention_mask, head_mask)
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


def gumbel_bert_model_forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    if attention_mask is None:
        # attention_mask = torch.ones_like(input_ids)  # NOTE
        attention_mask = torch.ones_like(input_ids[:, :, 0], dtype=torch.long)    # NOTE
    if token_type_ids is None:
        # token_type_ids = torch.zeros_like(input_ids)  # NOTE
        token_type_ids = torch.zeros_like(input_ids[:, :, 0], dtype=torch.long)  # NOTE

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
    encoder_outputs = self.encoder(embedding_output,
                                    extended_attention_mask,
                                    head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


def gumbel_bert_embedding_forward(self, input_ids, token_type_ids=None, position_ids=None):
    seq_length = input_ids.size(1)
    if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # NOTE
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])  # NOTE
    if token_type_ids is None:
        # token_type_ids = torch.zeros_like(input_ids)  # NOTE
        token_type_ids = torch.zeros_like(input_ids[:, :, 0])  # NOTE

    # words_embeddings = self.word_embeddings(input_ids)  # NOTE
    one_hots = input_ids.view(-1, input_ids.size(-1))
    embeddings = torch.matmul(one_hots, self.word_embeddings.weight)
    words_embeddings = embeddings.view(input_ids.size(0), input_ids.size(1), -1)
    # NOTE

    position_embeddings = self.position_embeddings(position_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = words_embeddings + position_embeddings + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


def raw_bert_model_forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
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
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


def raw_bert_embedding_forward(self, input_ids, token_type_ids=None, position_ids=None):
    seq_length = input_ids.size(1)
    if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)

    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = words_embeddings + position_embeddings + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class Decoder(torch.nn.Module):

    def __init__(self, pretrain_dir):
        super().__init__()
        config = transformers.BertConfig()
        config.num_hidden_layers = 6
        self.masked_lm = transformers.BertForMaskedLM(config)
        for layer in self.masked_lm.bert.encoder.layer:
            layer.cross_atten = transformers.modeling_bert.BertAttention(self.masked_lm.config)
            layer.cross_atten.load_state_dict(layer.attention.state_dict()) 

    def forward(self, input_ids, encoder_hidden):
        self.bind_methods()
        return self.masked_lm(input_ids, encoder_hidden)[0]

    def bind_methods(self):
        r"""Change forward method to add `encoder_hidden`.

        Architecture:
            (masked_lm): BertForMaskedLM              [change forward]
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
                    (intermediate): BertIntermediate
                    (output): BertOutput
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
            layer.attention.self.forward = types.MethodType(mask_bert_self_atten_forward, layer.attention.self)
    

class Seq2Seq(torch.nn.Module):

    def __init__(self, pretrain_dir, tokenizer):
        super(Seq2Seq, self).__init__()
        self.tokenizer = tokenizer
        # Encoder
        config = transformers.BertConfig()
        config.num_hidden_layers = 6
        self.encoder = transformers.BertModel(config)
        self.decoder = Decoder(pretrain_dir)
        # self.decoder.masked_lm.bert.embeddings = self.encoder.embeddings
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.decoder.masked_lm.resize_token_embeddings(len(self.tokenizer))

        # Decoder
        config = self.encoder.config
        self.decoder.masked_lm.cls.predictions.decoder = \
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.decoder.masked_lm.cls.predictions.decoder.weight = \
        #     self.encoder.embeddings.word_embeddings.weight
        self.decoder.masked_lm.cls.predictions.decoder.weight.data.copy_(
            self.encoder.embeddings.word_embeddings.weight)
        self.decoder.masked_lm.cls.predictions.bias = \
            torch.nn.Parameter(torch.zeros(config.vocab_size))
    
    def forward(self, *, mode, **kwargs):
        if mode == 'train':
            return self.forward_train(**kwargs)
        elif mode == 'valid':
            return self.forward_valid(**kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def forward_train(self, x, y):
        token_type_ids = self.get_token_type_ids(x)
        encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]
        logits = self.decoder(y[:, :-1], encoder_hidden)
        return logits

    def get_token_type_ids(self, x):
        token_type_ids = torch.zeros_like(x)
        for i in range(x.size(0)):
            sep_index = (x[i] == self.tokenizer.sep_token_id).nonzero()
            sep_index = sep_index.squeeze(-1).tolist()
            sep_index.append(len(x[0]))
            sep_index.append(len(x[i]) - 1)
            for j in range(0, len(sep_index) // 2 * 2, 2):
                start, end = sep_index[j], sep_index[j + 1]
                token_type_ids[i, start+1:end+1] = 1
        return token_type_ids

    def forward_valid(self, x, max_len):
        with torch.no_grad():
            token_type_ids = self.get_token_type_ids(x)
            encoder_hidden = self.encoder(x)[0]
        token_ids = torch.empty(x.size(0), max_len, dtype=torch.int64)
        token_ids[:, 0].fill_(self.tokenizer.bos_token_id)  # begin
        token_ids[:, 1:].fill_(self.tokenizer.pad_token_id)  # <pad>
        token_ids = token_ids.to(next(self.parameters()).device)
        for i in range(max_len - 1):
            with torch.no_grad():
                logits = self.decoder(token_ids, encoder_hidden)
            new_token_ids = logits[:, i].argmax(dim=-1)
            token_ids[:, i + 1] = new_token_ids
        return token_ids


class Seq2SeqKeywords(torch.nn.Module):

    def __init__(self, pretrain_dir, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        # X encoder (Bert, 6 layers)
        config = transformers.BertConfig()
        config.num_hidden_layers = 6
        self.encoder = transformers.BertModel(config)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        # Keywords decoder (Bert with cross attention, 6 layers)
        self.kw_decoder = Decoder(pretrain_dir)
        self.kw_decoder.masked_lm.resize_token_embeddings(len(self.tokenizer))
        config = self.encoder.config
        self.kw_decoder.masked_lm.cls.predictions.decoder = \
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.kw_decoder.masked_lm.cls.predictions.decoder.weight.data.copy_(
            self.encoder.embeddings.word_embeddings.weight)
        self.kw_decoder.masked_lm.cls.predictions.bias = \
            torch.nn.Parameter(torch.zeros(config.vocab_size))
        # NOTE: mean kw decoder
        # config = transformers.BertConfig()
        # config.num_hidden_layers = 6
        # self.kw_decoder = transformers.BertModel(config)
        # self.kw_linear = torch.nn.Linear(config.hidden_size, len(self.tokenizer))
        # NOTE

        # Keywords encoder (same as X encoder)
        config = transformers.BertConfig()
        config.num_hidden_layers = 6
        self.kw_encoder = transformers.BertModel(config)
        self.kw_encoder.resize_token_embeddings(len(self.tokenizer))

        # Y decoder (Bert with cross attention, 6 layers)
        # self.decoder = ResponseDecoder(pretrain_dir)
        # NOTE cross atten Decoder
        self.decoder = Decoder(pretrain_dir)
        # NOTE
        self.decoder.masked_lm.resize_token_embeddings(len(self.tokenizer))
        config = self.encoder.config
        self.decoder.masked_lm.cls.predictions.decoder = \
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.masked_lm.cls.predictions.decoder.weight.data.copy_(
            self.encoder.embeddings.word_embeddings.weight)
        self.decoder.masked_lm.cls.predictions.bias = \
            torch.nn.Parameter(torch.zeros(config.vocab_size))
    
    def forward(self, *, mode, **kwargs):
        if mode == 'train':
            return self.forward_train(**kwargs)
        elif mode == 'valid':
            return self.forward_valid(**kwargs)
        elif mode == 'valid_ppl':
            return self.forward_valid_ppl(**kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def forward_train(self, x, y, k, gt_kw_prob):
        # Step 1. X encoder
        token_type_ids = self.get_token_type_ids(x)
        encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]

        # Step 2. Keywords decoder
        kw_logits = self.kw_decoder(k[:, :-1], encoder_hidden)
        if random.random() > gt_kw_prob:
            # Gumbel then Mask
            one_hots = torch.nn.functional.gumbel_softmax(kw_logits, tau=1, hard=True)
            mask = torch.ones_like(one_hots)
            pad_mask = torch.zeros_like(one_hots)

            sep_one_hot = torch.zeros(one_hots.shape[-1])
            sep_one_hot[self.tokenizer.sep_token_id] = 1
            sep_one_hot = sep_one_hot.to(one_hots.device)

            batch_sep = (one_hots == sep_one_hot).sum(dim=-1) == one_hots.shape[-1]
            for i, row_sep in enumerate(batch_sep): 
                row_pos = row_sep.nonzero()
                if len(row_pos) > 0:
                    pos = row_pos[0, 0].item()
                    mask[i, pos+1:] = 0
                    pad_mask[i, pos+1:, self.tokenizer.pad_token_id] = 1
            one_hots = one_hots * mask + pad_mask
            # NOTE: detach sample
            # detach_kw_logits = kw_logits.clone().detach()
            # kw_preds = detach_kw_logits.argmax(dim=-1)
            # for i, row in enumerate(kw_preds == self.tokenizer.sep_token_id):
            #     pos = row.nonzero().squeeze(1).tolist()
            #     if len(pos) > 0:
            #         kw_preds[i, pos[0]+1:] = 0
            # NOTE
            # NOTE: mean decoder
            # kw_hiddden = self.kw_decoder(encoder_hidden)
            # kw_hidden = kw_hiddden.mean(dim=1)
            # kw_logits = self.kw_linear(kw_hiddden)
            # NOTE

            # Step 3. keywords encoder
            self.kw_encoder.forward = types.MethodType(gumbel_bert_model_forward, self.kw_encoder)
            self.kw_encoder.embeddings.forward = types.MethodType(gumbel_bert_embedding_forward, self.kw_encoder.embeddings)
            kw_hidden = self.kw_encoder(one_hots)[0]
        else:
            # NOTE: detach sample NOTE: with teach
            self.kw_encoder.forward = types.MethodType(raw_bert_model_forward, self.kw_encoder)
            self.kw_encoder.embeddings.forward = types.MethodType(raw_bert_embedding_forward, self.kw_encoder.embeddings)
            kw_hidden = self.kw_encoder(k)[0]
            # NOTE

        # Step 4. Y decoder
        # NOTE
        hidden = torch.cat([encoder_hidden, kw_hidden], dim=1)
        rsp_logits = self.decoder(y[:, :-1], hidden)
        # NOTE
        # rsp_logits = self.decoder(y[:, :-1], kw_hidden, encoder_hidden)    
        return rsp_logits, kw_logits

    def get_token_type_ids(self, x):
        token_type_ids = torch.zeros_like(x)
        for i in range(x.size(0)):
            sep_index = (x[i] == self.tokenizer.sep_token_id).nonzero()
            sep_index = sep_index.squeeze(-1).tolist()
            sep_index.append(len(x[0]))
            sep_index.append(len(x[i]) - 1)
            for j in range(0, len(sep_index) // 2 * 2, 2):
                start, end = sep_index[j], sep_index[j + 1]
                token_type_ids[i, start+1:end+1] = 1
        return token_type_ids

    def forward_valid(self, x, k, max_len):
        # Gumbel
        self.kw_encoder.forward = types.MethodType(raw_bert_model_forward, self.kw_encoder)
        self.kw_encoder.embeddings.forward = types.MethodType(raw_bert_embedding_forward, self.kw_encoder.embeddings)

        # Step 1. X encoder
        with torch.no_grad():
            token_type_ids = self.get_token_type_ids(x)
            encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]

        if k is None:
            # Step 2. Keywords decoder
            kw_prediction = torch.empty(x.size(0), max_len, dtype=torch.int64)
            kw_prediction[:, 0].fill_(self.tokenizer.bos_token_id)
            kw_prediction[:, 1:].fill_(self.tokenizer.pad_token_id)
            kw_prediction = kw_prediction.to(next(self.parameters()).device)
            for i in range(max_len - 1):
                with torch.no_grad():
                    kw_logits = self.kw_decoder(kw_prediction, encoder_hidden)
                new_kw_prediction = kw_logits[:, i].argmax(dim=-1)
                kw_prediction[:, i + 1] = new_kw_prediction

            # Step 3. Keywords encoder
            with torch.no_grad():
                kw_hidden = self.kw_encoder(kw_prediction)[0]
        else:
            with torch.no_grad():
                kw_hidden = self.kw_encoder(k)[0]

        # Step 4. Y decoder
        # NOTE
        hidden = torch.cat([encoder_hidden, kw_hidden], dim=1)
        # NOTE
        y_prediction = torch.empty(x.size(0), max_len, dtype=torch.int64)
        y_prediction[:, 0].fill_(self.tokenizer.bos_token_id)
        y_prediction[:, 1:].fill_(self.tokenizer.pad_token_id)
        y_prediction = y_prediction.to(next(self.parameters()).device)
        for i in range(max_len - 1):
            with torch.no_grad():
                # logits = self.decoder(y_prediction, kw_hidden, encoder_hidden)
                # NOTE
                logits = self.decoder(y_prediction, hidden)
            new_y_prediction = logits[:, i].argmax(dim=-1)
            y_prediction[:, i + 1] = new_y_prediction

        if k is None:
            return y_prediction, kw_prediction
        else:
            return y_prediction

    def forward_valid_ppl(self, x, k, y, max_len):
        # Gumbel
        self.kw_encoder.forward = types.MethodType(raw_bert_model_forward, self.kw_encoder)
        self.kw_encoder.embeddings.forward = types.MethodType(raw_bert_embedding_forward, self.kw_encoder.embeddings)

        # Step 1. X encoder
        with torch.no_grad():
            token_type_ids = self.get_token_type_ids(x)
            encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]

        if k is None:
            # Step 2. Keywords decoder
            kw_prediction = torch.empty(x.size(0), max_len, dtype=torch.int64)
            kw_prediction[:, 0].fill_(self.tokenizer.bos_token_id)
            kw_prediction[:, 1:].fill_(self.tokenizer.pad_token_id)
            kw_prediction = kw_prediction.to(next(self.parameters()).device)
            for i in range(max_len - 1):
                with torch.no_grad():
                    kw_logits = self.kw_decoder(kw_prediction, encoder_hidden)
                new_kw_prediction = kw_logits[:, i].argmax(dim=-1)
                kw_prediction[:, i + 1] = new_kw_prediction

            # Step 3. Keywords encoder
            with torch.no_grad():
                kw_hidden = self.kw_encoder(kw_prediction)[0]
        else:
            with torch.no_grad():
                kw_hidden = self.kw_encoder(k)[0]

        # Step 4. Y decoder
        hidden = torch.cat([encoder_hidden, kw_hidden], dim=1)
        rsp_logits = self.decoder(y[:, :-1], hidden)

        return rsp_logits
