import types

import torch
from transformers import BertConfig, BertForMaskedLM, BertModel

BERT_CONFIG = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 30522,
    "is_decoder": False,
    "output_past": False,  # For generation
}


def get_token_type_ids(x, sep_token_id):  # TODO
    token_type_ids = torch.zeros_like(x)
    for i in range(x.size(0)):
        sep_index = (x[i] == sep_token_id).nonzero()
        sep_index = sep_index.squeeze(-1).tolist()
        sep_index.append(len(x[0]))
        sep_index.append(len(x[i]) - 1)
        for j in range(0, len(sep_index) // 2 * 2, 2):
            start, end = sep_index[j], sep_index[j + 1]
            token_type_ids[i, start + 1:end + 1] = 1
    return token_type_ids


class Seq2Seq(torch.nn.Module):

    def __init__(self, tokenizer, max_decode_len):
        super(Seq2Seq, self).__init__()
        self.tokenizer = tokenizer
        self.max_decode_len = max_decode_len

        self.enc_config = BertConfig(**BERT_CONFIG)
        self.enc_config.vocab_size = len(self.tokenizer)
        self.encoder = BertModel(config=self.enc_config)

        self.dec_config = BertConfig(**BERT_CONFIG)
        self.dec_config.is_decoder = True
        self.dec_config.vocab_size = len(self.tokenizer)
        self.decoder = BertForMaskedLM(config=self.dec_config)

        self.prepare_generation()

    def load_pretrain(self, path):
        self.enc_config = BertConfig(**BERT_CONFIG)
        self.encoder = BertModel.from_pretrained(path, config=self.enc_config)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.dec_config = BertConfig(**BERT_CONFIG)
        self.dec_config.is_decoder = True
        self.decoder = BertForMaskedLM.from_pretrained(path, config=self.dec_config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        for layer in self.decoder.bert.encoder.layer:
            state_dict = layer.attention.state_dict()
            layer.crossattention.load_state_dict(state_dict)

        self.prepare_generation()

    def prepare_generation(self):
        def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
            return {
                'input_ids': input_ids,
                'encoder_hidden_states': past,
            }
        self.decoder.prepare_inputs_for_generation = \
            types.MethodType(prepare_inputs_for_generation, self.decoder)

    def forward(self, *, mode, **kwargs):
        assert mode in ['train', 'test']
        if mode == 'train':
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, x, y):
        token_type_ids = self.get_token_type_ids(x)
        encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]
        logits = self.decoder(y[:, :-1], encoder_hidden_states=encoder_hidden)[0]
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
                token_type_ids[i, start + 1:end + 1] = 1
        return token_type_ids

    @torch.no_grad()
    def forward_test(self, x):
        token_type_ids = self.get_token_type_ids(x)
        encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]
        return self.generate(encoder_hidden, x.size(0))

    @torch.no_grad()
    def generate(self, encoder_hidden, batch_size):
        input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=next(self.decoder.parameters()).device,
        )
        return self.decoder._generate_no_beam_search(
            input_ids,
            cur_len=1,
            max_length=self.max_decode_len,
            min_length=0,
            do_sample=False,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.pad_token_id,
            batch_size=batch_size,
            encoder_outputs=encoder_hidden,
            attention_mask=input_ids.new_ones(input_ids.shape),
        )


class KWSeq2Seq(torch.nn.Module):

    def __init__(self, tokenizer, max_decode_len, pretrain_path=None):
        super(Seq2Seq, self).__init__()
        self.tokenizer = tokenizer
        self.max_decode_len = max_decode_len
        self.pretrain_path = pretrain_path

        if self.pretrain_path is None:
            self.init_from_scratch()
        else:
            self.init_from_pretrain()

        self.prepare_generation()

    def init_from_scratch(self):
        # Encoder
        self.enc_config = BertConfig(**BERT_CONFIG)
        self.enc_config.num_hidden_layers = 6
        self.enc_config.vocab_size = len(self.tokenizer)
        self.encoder = BertModel(config=self.enc_config)

        # Decoder
        self.dec_config = BertConfig(**BERT_CONFIG)
        self.dec_config.num_hidden_layers = 6
        self.dec_config.is_decoder = True
        self.dec_config.vocab_size = len(self.tokenizer)
        self.decoder = BertForMaskedLM(config=self.dec_config)

        # KW-Encoder
        self.kw_enc_config = BertConfig(**BERT_CONFIG)
        self.kw_enc_config.num_hidden_layers = 6
        self.kw_enc_config.vocab_size = len(self.tokenizer)
        self.kw_encoder = BertModel(config=self.kw_enc_config)

        # KW-Decoder
        self.kw_dec_config = BertConfig(**BERT_CONFIG)
        self.kw_dec_config.num_hidden_layers = 6
        self.kw_dec_config.is_decoder = True
        self.kw_dec_config.vocab_size = len(self.tokenizer)
        self.kw_decoder = BertForMaskedLM(config=self.kw_dec_config)

    def init_from_pretrain(self):
        # Encoder
        self.enc_config = BertConfig(**BERT_CONFIG)
        self.enc_config.num_hidden_layers = 6
        self.encoder = BertModel.from_pretrained(
            self.pretrain_path, config=self.enc_config)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        # Decoder
        self.dec_config = BertConfig(**BERT_CONFIG)
        self.dec_config.num_hidden_layers = 6
        self.dec_config.is_decoder = True
        self.decoder = BertForMaskedLM.from_pretrained(
            self.pretrain_path, config=self.dec_config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        for layer in self.decoder.bert.encoder.layer:
            state_dict = layer.attention.state_dict()
            layer.crossattention.load_state_dict(state_dict)

        # KW-Encoder
        self.kw_enc_config = BertConfig(**BERT_CONFIG)
        self.kw_enc_config.num_hidden_layers = 6
        self.kw_encoder = BertModel.from_pretrained(
            self.pretrain_path, config=self.kw_enc_config)
        self.kw_encoder.resize_token_embeddings(len(self.tokenizer))

        # KW-Decoder
        self.kw_dec_config = BertConfig(**BERT_CONFIG)
        self.kw_dec_config.num_hidden_layers = 6
        self.kw_dec_config.is_decoder = True
        self.kw_decoder = BertForMaskedLM.from_pretrained(
            self.pretrain_path, config=self.kw_dec_config)
        self.kw_decoder.resize_token_embeddings(len(self.tokenizer))
        for layer in self.kw_decoder.bert.encoder.layer:
            state_dict = layer.attention.state_dict()
            layer.crossattention.load_state_dict(state_dict)

    def prepare_generation(self):
        def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
            return {
                'input_ids': input_ids,
                'encoder_hidden_states': past,
            }
        self.decoder.prepare_inputs_for_generation = \
            types.MethodType(prepare_inputs_for_generation, self.decoder)
        self.kw_decoder.prepare_inputs_for_generation = \
            types.MethodType(prepare_inputs_for_generation, self.kw_decoder)

    def forward(self, *, mode, **kwargs):
        assert mode in ['train', 'test']
        if mode == 'train':
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, x, y, k):
        token_type_ids = get_token_type_ids(x, self.tokenizer.sep_token_id)
        encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]

        kw_logits = self.kw_decoder(k[:, :-1], encoder_hidden_states=encoder_hidden)[0]

        kw_token_type_ids = get_token_type_ids(k, self.tokenizer.sep_token_id)
        kw_encoder_hidden = self.kw_encoder(k, token_type_ids=kw_token_type_ids)[0]

        hidden = torch.cat([encoder_hidden, kw_encoder_hidden], dim=1)
        y_logits = self.decoder(y[:, :-1], encoder_hidden_states=hidden)[0]

        return y_logits, kw_logits

    @torch.no_grad()
    def forward_test(self, x):
        token_type_ids = self.get_token_type_ids(x)
        encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]

        kw_ids = self.generate(self.kw_decoder, encoder_hidden, x.size(0))

        kw_token_type_ids = self.get_token_type_ids(kw_ids)
        kw_encoder_hidden = self.kw_encoder(kw_ids, token_type_ids=kw_token_type_ids)[0]

        hidden = torch.cat([encoder_hidden, kw_encoder_hidden], dim=1)
        y_ids = self.generate(self.decoder, hidden, x.size(0))

        return y_ids, kw_ids

    @torch.no_grad()
    def generate(self, decoder, encoder_hidden, batch_size):
        input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=next(decoder.parameters()).device,
        )
        return decoder._generate_no_beam_search(
            input_ids,
            cur_len=1,
            max_length=self.max_decode_len,
            min_length=0,
            do_sample=False,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.pad_token_id,
            batch_size=batch_size,
            encoder_outputs=encoder_hidden,
            attention_mask=input_ids.new_ones(input_ids.shape),
        )
