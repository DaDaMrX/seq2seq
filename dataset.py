import argparse
import logging
import pickle
import random
import re

import torch
from tqdm import tqdm
from transformers import BertTokenizer


class Data(dict):

    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        if attr in self.keys():
            return self.__getitem__(attr)
        else:
            super().__getattr__(attr)

    def size(self):
        attrs = ['x', 'y', 'k']
        return sum(len(self[key]) for key in attrs if key in self)

    def to(self, device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)


class DailyDataset:

    def __init__(self, path, tokenizer, max_x_len=500,
                 context_size=5, keywords_path=None):
        self.path = path
        self.tokenizer = tokenizer
        self.max_x_len = max_x_len
        self.context_size = context_size
        self.keywords_path = keywords_path
        self.logger = self.get_logger('DailyDataset')

        convs = self.load()
        convs = self.clean(convs)
        self.data = self.make_pairs(convs)
        if keywords_path is not None:
            self.data = self.load_keywords()
        self.examples = self.make_examples()

    def get_logger(self, class_name):
        self.logger = logging.getLogger(__name__ + '.' + class_name)
        self.logger.setLevel(logging.INFO)
        hander = logging.StreamHandler()
        hander.setLevel(logging.INFO)
        formatter = logging.Formatter(
            f'[%(levelname)s] [{class_name}.%(funcName)s] %(message)s')
        hander.setFormatter(formatter)
        self.logger.addHandler(hander)
        return self.logger

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def load(self):
        self.logger.info(f'Loading data from "{self.path}"')
        with open(self.path) as f:
            lines = f.readlines()
        convs = [line.split('__eou__')[:-1] for line in lines]
        self.logger.info(f'{len(convs)} conversations loaded.')
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
        self.logger.info(f'{len(convs_clean)} conversations cleaned.')
        return convs_clean

    def make_pairs(self, convs):
        data = []
        for conv in convs:
            for i in range(1, len(conv)):
                context = conv[max(i - self.context_size, 0):i]
                data.append((context, conv[i]))
        self.logger.info(f'{len(data)} pairs.')
        return data

    def load_keywords(self):
        with open(self.keywords_path) as f:
            lines = f.readlines()
        keywords = [line.strip().split() for line in lines]
        assert len(self.data) == len(keywords)

        data_keywords = []
        for (x, y), words in zip(self.data, keywords):
            data_keywords.append((x, y, words))
        self.data = data_keywords
        self.logger.info(f'{len(self.data)} pairs with keywords.')
        return self.data

    def make_examples(self):
        self.examples = []
        for index, item in enumerate(tqdm(self.data, desc='Make Examples')):
            self.examples.append(self.make_example(index, item))
        return self.examples

    def make_example(self, index, item):
        example = Data(index=index)

        context, text_y = item[0], item[1]
        example.texts_x = context
        text_y = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text_y))
        example.text_y = text_y

        example.x = []
        for sent in reversed(context):
            tokens = self.tokenizer.tokenize(sent)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(example.x) + len(ids) + 2 <= self.max_x_len:
                example.x = ids + [self.tokenizer.sep_token_id] + example.x
            else:
                break
        example.x = [self.tokenizer.cls_token_id] + example.x

        tokens = self.tokenizer.tokenize(text_y)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        example.y = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.sep_token_id]

        if len(item) > 2:
            example.tokens_k = item[2]
            k = self.tokenizer.convert_tokens_to_ids(item[2])
            example.k = [self.tokenizer.bos_token_id] + k + [self.tokenizer.sep_token_id]
        return example


class DataLoader:

    def __init__(self, examples, max_tokens, pad_value=0):
        self.examples = examples
        self.max_tokens = max_tokens
        self.pad_value = pad_value
        self.logger = self.get_logger('DataLoader')

        # Sort & filter
        random.shuffle(self.examples)
        self.examples.sort(key=lambda e: len(e.x) + len(e.y))
        while self.examples[-1].size() > self.max_tokens:
            self.examples.pop()

        self.batches = self.make_batches()

    def get_logger(self, class_name):
        self.logger = logging.getLogger(__name__ + '.' + class_name)
        self.logger.setLevel(logging.INFO)
        hander = logging.StreamHandler()
        hander.setLevel(logging.INFO)
        formatter = logging.Formatter(
            f'[%(levelname)s] [{class_name}.%(funcName)s] %(message)s')
        hander.setFormatter(formatter)
        self.logger.addHandler(hander)
        return self.logger

    def dump(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.batches, f)
        self.logger.info(f'Dump to "{pickle_path}"')

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)

    def make_batches(self):
        # Chunk of size `self.max_tokens` by Two pointers (left & right)
        self.batches = []
        left, size = 0, 0
        for right, example in enumerate(tqdm(self.examples, desc='Make Batches')):
            size += example.size()
            if size > self.max_tokens:
                self.batches.append(self.make_batch(self.examples[left:right]))
                left, size = right, example.size()
        self.batches.append(self.make_batch(self.examples[left:]))

        self.logger.info(f'{len(self.batches)} Batches')
        return self.batches

    def make_batch(self, examples):
        # Zip
        batch = Data(**{key: [] for key in examples[0].keys()})
        for e in examples:
            for k, v in batch.items():
                v.append(e[k])

        # Pad
        def pad_sequence(sequence):
            return torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in sequence],
                batch_first=True,
                padding_value=self.pad_value,
            )

        batch.x = pad_sequence(batch.x)
        batch.y = pad_sequence(batch.y)
        if hasattr(batch, 'k'):
            batch.k = pad_sequence(batch.k)

        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', required=True)
    parser.add_argument('--kw_path', '-k', default=None)
    parser.add_argument('--batch_size', '-b', default=2500, type=int)
    parser.add_argument('--pickle_path', '-p', default=None)
    parser.add_argument('--max_x_len', default=500, type=int)
    parser.add_argument('--context_size', default=5, type=int)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('pretrain/vocab.txt')
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})

    dataset = DailyDataset(
        path=args.data_path,
        tokenizer=tokenizer,
        keywords_path=args.kw_path,
        max_x_len=args.max_x_len,
        context_size=args.context_size,
    )
    dataloader = DataLoader(
        examples=dataset.examples,
        max_tokens=args.batch_size,
        pad_value=tokenizer.pad_token_id,
    )
    dataloader.dump(args.pickle_path)

    batch = dataloader.batches[100]
    for k, v in batch.items():
        print(f'    {k}: {type(v)}')


"""
python dataset.py --data_path=data/dialogues_test.txt \
    --kw_path=data/keywords_test.txt \
    --batch_size=5000 \
    --pickle_path=pickle/daily_test_5000.pickle

python dataset.py --data_path=data/dialogues_train.txt \
    --kw_path=data/keywords_train.txt \
    --batch_size=3000 \
    --pickle_path=pickle/daily_train_3000.pickle
"""
