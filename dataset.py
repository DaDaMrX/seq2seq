import argparse
import logging
import pickle
import re

import torch
from tqdm import tqdm
from transformers import BertTokenizer


class Example:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def size(self):
        num = 0
        if hasattr(self, 'x'):
            num += len(self.x)
        if hasattr(self, 'y'):
            num += len(self.y)
        if hasattr(self, 'k'):
            num += len(self.k)
        return num

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
                text += f'{type(v)}'
            text = '    ' + text + ','
            items.append(text)
        items = '\n'.join(items)
        return f'Example(\n{items}\n)'


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
        self.data.sort(key=lambda t: sum(map(len, t[0])) + len(t[1]))
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
        for d in tqdm(self.data, desc='Making Examples', dynamic_ncols=True):
            self.examples.append(self.make_example(d))
        return self.examples

    def make_example(self, item):
        context, text_y = item[0], item[1]
        text_y = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text_y))
        example = Example(texts_x=context, text_y=text_y)

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


class Batch:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)

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


class DataLoader:

    def __init__(self, dataset, max_tokens=2500):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.logger = self.get_logger('DataLoader')
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
        batches = []
        examples, num = [], 0
        for example in tqdm(self.dataset, desc='Batching', dynamic_ncols=True):
            num += example.size()
            examples.append(example)
            if num == self.max_tokens:
                batches.append(self.make_batch(examples))
                examples, num = [], 0
            elif num > self.max_tokens:
                batches.append(self.make_batch(examples[:-1]))
                examples, num = [example], example.size()
        self.logger.info(f'{len(batches)} batches.')
        return batches

    def make_batch(self, examples):
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
        if has_keywords:
            batch_k = torch.nn.utils.rnn.pad_sequence(
                batch_k, batch_first=True, padding_value=padding_value)
            batch.batch_k = batch_k
            batch.batch_tokens_k = batch_tokens_k
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/')
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})

    dataset = DailyDataset(
        path=args.data_path,
        tokenizer=tokenizer,
        keywords_path=args.kw_path,
        max_x_len=args.max_x_len,
        context_size=args.context_size,
    )
    dataloader = DataLoader(
        dataset=dataset,
        max_tokens=args.batch_size,
    )
    dataloader.dump(args.pickle_path)


"""
python dataset.py --data_path=data/dialogues_test.txt \
    --kw_path=data/keywords_test.txt \
    --batch_size=3000 \
    --pickle_path=daily_test_3000.pickle

python dataset.py --data_path=data/dialogues_train.txt \
    --kw_path=data/keywords_train.txt \
    --batch_size=2500 \
    --pickle_path=daily_train_2500.pickle
"""
