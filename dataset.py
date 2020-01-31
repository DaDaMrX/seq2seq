import argparse
import re
import os
import pickle
import random

import torch
from tqdm import tqdm


class Example:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = []
        for k, v in self.__dict__.items():
            text = f'{k}: '
            if isinstance(v, list):
                text += f'List[{type(v[0])}], len={len(v)}'
            elif isinstance(v, tuple):
                text += f'Tuple[{type(v[0])}], len={len(v)}'
            elif isinstance(v, str):
                text += f'str: {v}'
            else:
                text += f'type={type(v)}, {v}'
            text = '    ' + text + ','
            items.append(text)
        items = '\n'.join(items) 
        return f'Example(\n{items}\n)'


class DailyDataset:

    def __init__(self, path, tokenizer, max_len, keywords_path=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        print('Load data from', path)
        convs = self.load(path)
        convs = self.clean(convs)
        print(f'{len(convs)} convs')
        self.data = []
        for conv in convs:
            for i in range(1, len(conv)):
                self.data.append((conv[max(i-5,0):i], conv[i]))
        print(len(self.data), 'pairs')

        self.has_keywords = False
        if keywords_path is not None:
            self.has_keywords = True
            with open(keywords_path) as f:
                lines = f.readlines()
            self.keywords = [line.strip().split() for line in lines]
            assert len(self.data) == len(self.keywords)

            data_keywords = []
            for (x, y), words in zip(self.data, self.keywords):
                data_keywords.append((x, y, words))
            self.data = data_keywords
            print(f'{len(self.data)} with keywords')

        self.data.sort(key=lambda d: sum(map(len, d[0])) + len(d[1]))

    def load(self, path):
        with open(path) as f:
            lines = f.readlines()
        convs = [line.split('__eou__')[:-1] for line in lines]
        return convs

    def clean(self, convs):
        def _clean(s):
            s = s.strip().lower()
            s = re.sub(r'(\w)\.(\w)', r'\1 . \2', s)
            s = s.replace('。', '.')
            s = s.replace(';', ',')
            s = s.replace('’', "'")
            s = s.replace(' p . m . ', ' pm ')
            s = s.replace(' a . m . ', ' am ')
            return s
        convs_clean = []
        for conv in convs:
            convs_clean.append(list(map(_clean, conv)))
        return convs_clean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.has_keywords:
            sents, y, k = self.data[index]
            y = tokenizer.convert_tokens_to_string(tokenizer.tokenize(y))
            example = Example(texts_x=sents, text_y=y, tokens_k=k)
        else:
            sents, y = self.data[index]
            y = tokenizer.convert_tokens_to_string(tokenizer.tokenize(y))
            example = Example(texts_x=sents, text_y=y)
        x = []
        for sent in reversed(sents):
            sent = self.tokenizer.encode(sent)
            if len(x) + len(sent) + 1 <= self.max_len:
                x = sent + [self.tokenizer.sep_token_id] + x
            else:
                break
        x = [self.tokenizer.cls_token_id] + x
        y = self.tokenizer.encode(y)
        y = [self.tokenizer.bos_token_id] + y + [self.tokenizer.sep_token_id]
        example.x = x
        example.y = y

        if self.has_keywords:
            k = self.tokenizer.convert_tokens_to_ids(k)
            # sep_k = [self.tokenizer.bos_token_id]
            # for x in k:
            #     sep_k.append(x)
            #     sep_k.append(self.tokenizer.sep_token_id)
            k = [self.tokenizer.bos_token_id] + k + [self.tokenizer.sep_token_id]
            example.k = sep_k
        return example


class Batch:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
            
    def __repr__(self):
        items = []
        for k, v in self.__dict__.items():
            text = f'{k}: '
            if isinstance(v, torch.Tensor):
                text += f'{v.shape}, {v.dtype}'
            elif isinstance(v, list):
                text += f'List[{type(v[0])}], len={len(v)}'
            elif isinstance(v, tuple):
                text += f'Tuple[{type(v[0])}], len={len(v)}'
            else:
                text += f'type={type(v)}, {v}'
            text = '    ' + text + ','
            items.append(text)
        items = '\n'.join(items) 
        return f'Batch(\n{items}\n)'

    def split(self, n):
        if n == 1:
            return [self]
        values = self.__dict__.values()
        batch_size = len(next(iter(values)))
        for value in values:
            assert len(value) == batch_size
        sub_batches = []
        sub_batch_size = (batch_size + n - 1) // n
        for start in range(0, batch_size, sub_batch_size):
            b = Batch()
            for k, v in self.__dict__.items():
                b.__dict__[k] = v[start:start+sub_batch_size]
            sub_batches.append(b)
        return sub_batches

    def size(self):
        return len(next(iter(self.__dict__.values())))


class DataLoader:

    def __init__(self, dataset, max_tokens, cache_path=None):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.batches = self.make_batches()
        if cache_path is not None:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.batches, f)
            print('Dump pickle to', cache_path)

    def make_batches(self):
        batches = []
        batch, tokens = [], 0
        for data in tqdm(self.dataset, desc='Batching'):
            # num = len(data[0]) + len(data[1])
            num = len(data.x) + len(data.y)
            if self.dataset.has_keywords:
                num += len(data.k)
            tokens += num
            batch.append(data)
            if tokens == self.max_tokens:
                batches.append(self.make_batch(batch))
                batch, tokens = [], 0
            elif tokens > self.max_tokens:
                batches.append(self.make_batch(batch[:-1]))
                batch, tokens = [data], num
        return batches

    def make_batch(self, examples):
        # if self.dataset.has_keywords:
        #     batch_x, batch_y, batch_k = list(zip(*batch))
        # else:
        #     batch_x, batch_y = list(zip(*batch))

        has_keywords = hasattr(examples[0], 'k')
        batch_x, batch_y = [], []
        batch_texts_x, batch_text_y = [], []
        if has_keywords:
            batch_k = []
            batch_tokens_k = []
        for example in examples:
            batch_x.append(torch.tensor(example.x))
            batch_y.append(torch.tensor(example.y))
            batch_texts_x.append(example.texts_x)
            batch_text_y.append(example.text_y)
            if has_keywords:
                batch_k.append(torch.tensor(example.k))
                batch_tokens_k.append(example.tokens_k)

        # batch_x = [torch.tensor(x) for x in batch_x]
        # batch_y = [torch.tensor(y) for y in batch_y]

        padding_value = self.dataset.tokenizer.pad_token_id
        batch_x = torch.nn.utils.rnn.pad_sequence(
            batch_x, batch_first=True, padding_value=padding_value)
        batch_y = torch.nn.utils.rnn.pad_sequence(
            batch_y, batch_first=True, padding_value=padding_value)

        batch = Batch(
            batch_x=batch_x, 
            batch_y=batch_y,
            batch_texts_x=batch_texts_x,
            batch_text_y=batch_text_y,
        )

        if self.dataset.has_keywords:
            # batch_k = [torch.tensor(k) for k in batch_k]
            batch_k = torch.nn.utils.rnn.pad_sequence(
                batch_k, batch_first=True, padding_value=padding_value)
            batch.batch_k = batch_k
            batch.batch_tokens_k = batch_tokens_k
        return batch


if __name__ == '__main__':
    import torch
    from transformers import BertTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', choices=['daily', 'cornell'], required=True)
    parser.add_argument('--split', '-s', choices=['train', 'valid', 'test'], required=True)
    parser.add_argument('--max_x_len', default=500, type=int)
    parser.add_argument('--max_batch_tokens', default=2500, type=int)
    parser.add_argument('--kw_path', '-k', default=None)
    parser.add_argument('--pickle_path', '-p', default=None)
    args = parser.parse_args()

    paths = {
        'daily': {
            'train': {
                'data_path': 'data/dialogues_train.txt',
                'keywords_path': 'data/keywords_train.txt',
                'pickle_path': 'data/dialogues_train_keywords_2500_sep.pickle',
            },
            'valid': {
                'data_path': 'data/dialogues_validation.txt',
                'keywords_path': 'data/keywords_validation.txt',
                'pickle_path': 'data/dialogues_validation_keywords_2500_sep.pickle',
            },
            'test': {
                'data_path': 'data/dialogues_test.txt',
                'keywords_path': 'data/keywords_test.txt',
                'pickle_path': 'data/dialogues_test_keywords_2500_sep.pickle',
            }
        },
        'cornell': {
            'train': {
                'data_path': 'cornellmovie_data/cornellmovie_train.txt',
                'keywords_path': 'cornellmovie_data/keywords_train.txt',
                'pickle_path': f'cornellmovie_data/cornellmovie_train_{args.max_batch_tokens}.pickle',
            },
            'valid': {
                'data_path': 'cornellmovie_data/cornellmovie_valid.txt',
                'keywords_path': 'cornellmovie_data/keywords_validation.txt',
                'pickle_path': f'cornellmovie_data/cornellmovie_valid_{args.max_batch_tokens}.pickle',
            },
            'test': {
                'data_path': 'cornellmovie_data/cornellmovie_test.txt',
                'keywords_path': 'cornellmovie_data/keywords_test.txt',
                'pickle_path': f'cornellmovie_data/cornellmovie_test_{args.max_batch_tokens}.pickle',
            }
        },
    }

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/')
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})

    kw_path = paths[args.dataset][args.split]['keywords_path'] \
        if args.kw_path is None else args.kw_path
    pickle_path = paths[args.dataset][args.split]['pickle_path'] \
        if args.pickle_path is None else args.pickle_path
    
    dataset = DailyDataset(
        path=paths[args.dataset][args.split]['data_path'],
        tokenizer=tokenizer,
        max_len=args.max_x_len,
        keywords_path=kw_path,
    )
    dataloader = DataLoader(
        dataset=dataset,
        max_tokens=args.max_batch_tokens,
        cache_path=pickle_path,
    )

    # with open(pickle_path, 'rb') as f:
    #     batches = pickle.load(f)
    # print(len(batches))
    # sizes = [b.size() for b in batches]
    # ave = sum(sizes) / len(sizes)
    # print(f'Average batch size: {ave:.3f}')
    # print(batches[100])
    # print(batches[100].batch_k[0])
