import argparse
import logging
import os
import pickle
import random
import shutil
import time

import pynvml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

from dataset import Batch
from evaluator import Evaluator
from model import KWSeq2Seq, Seq2Seq
from recoder import Recoder


class Trainer:

    def __init__(self, args):
        self.args = args
        self.logger = self.get_logger('Trainer')
        self.tensorboard_dir, self.checkpoints_dir = self.prepare_dirs()
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.recoder = Recoder(tag=self.args.tag,
                               clear=self.args.clear,
                               port=self.args.mongodb_port,
                               db_name=self.args.mongodb_db_name)

        self.device, self.device_ids = self.get_device()
        self.tokenizer = BertTokenizer(vocab_file=self.args.vocab_path)
        self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})

        model_args = dict(
            tokenizer=self.tokenizer,
            max_decode_len=self.args.max_decode_len,
        )
        if args.use_keywords:
            self.logger.info('Create KWSeq2Seq model...')
            self.model = KWSeq2Seq(**model_args)
        else:
            self.logger.info('Create Seq2Seq model...')
            self.model = Seq2Seq(**model_args)

        if self.args.resume:
            self.model = self.resume()
        elif self.args.pretrain_path is not None:
            self.logger.info(f'Load pretrain from "{self.args.pretrain_path}"...')
            self.model.load_pretrain(self.args.pretrain_path)

        self.logger.info(f'Moving model to {self.device}...')
        self.model = self.model.to(self.device)
        self.logger.info(f'Moving model to {self.device} done.')
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def get_device(self):
        if args.n_gpu <= 0:
            self.device = torch.device('cpu')
            self.device_ids = []
        else:
            info = self.get_gpu_info()[:self.args.n_gpu]
            self.device = torch.device(info[0]['id'])
            self.device_ids = [item['id'] for item in info]
            for item in info:
                self.logger.info(f'Use GPU {item["id"]} (free: {item["free"]} MB)')
        return self.device, self.device_ids

    def get_gpu_info(self):
        pynvml.nvmlInit()
        n_cards = pynvml.nvmlDeviceGetCount()
        mb = 1024 * 1024
        info = []
        for index in range(n_cards):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info.append({
                'id': index,
                'free': memory.free // mb,
                'used': memory.used // mb,
                'total': memory.total // mb,
                'ratio': memory.used / memory.total,
            })
        info.sort(key=lambda x: x['free'], reverse=True)
        return info

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

    def prepare_dirs(self):
        self.tensorboard_dir = os.path.join(
            self.args.tensorboard_base_dir, self.args.tag)
        self.checkpoints_dir = os.path.join(
            self.args.checkpoints_base_dir, self.args.tag)

        for d in [self.tensorboard_dir, self.checkpoints_dir]:
            if os.path.exists(d):
                if self.args.clear:
                    self.logger.info(f'Clear "{d}"')
                    shutil.rmtree(d)
                    time.sleep(1)
                elif self.args.resume:
                    continue
                else:
                    raise RuntimeError(f'"{d}" already exsits.')
            os.makedirs(d)

        return self.tensorboard_dir, self.checkpoints_dir

    def resume(self):
        ckpts = {int(os.path.splitext(name)[0].split('-')[-1]): name
                 for name in os.listdir(self.checkpoints_dir)}
        assert len(ckpts) > 0
        self.epoch = max(ckpts)
        latest_path = os.path.join(self.checkpoints_dir, ckpts[self.epoch])
        self.logger.info(f'Load checkpoint from "{latest_path}"...')
        state_dict = torch.load(latest_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        return self.model

    def loss_fn(self, input, target):
        loss = torch.nn.functional.cross_entropy(
            input=input.reshape(-1, input.size(-1)),
            target=target[:, 1:].reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction='mean',
        )
        return loss

    def overfit(self, n_steps):
        with open(self.args.train_pickle_path, 'rb') as f:
            self.train_batches = pickle.load(f)
        batch = self.train_batches[100]
        batch.to(self.device)

        self.model.train().zero_grad()
        self.epoch, self.train_steps = 0, 0
        pbar = tqdm(range(n_steps), desc='Overfit', dynamic_ncols=True)
        for i in pbar:
            loss = self.train_batch(batch)
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        self.test_steps = 0
        pred = self.test_batch(batch)
        for k, v in pred.items():
            print(f'{k}: {v[0]:.4f}')

        self.evaluator = Evaluator(batches=[batch])
        scores = self.evaluator.evaluate(pred, epoch=0)
        for k, v in scores.items():
            print(f'{k}: {v}')

    def train_batch(self, batch):
        # Forward & Loss
        batch.to(self.device)
        if self.args.use_keywords:
            gt_kw_prob = 1
            rsp_logits, kw_logits = self.model(
                mode='train', x=batch.batch_x, y=batch.batch_y,
                k=batch.batch_k, gt_kw_prob=gt_kw_prob)
            rsp_loss = self.loss_fn(input=rsp_logits, target=batch.batch_y)
            kw_loss = self.loss_fn(input=kw_logits, target=batch.batch_k)
            loss = self.args.response_loss_weight * rsp_loss + \
                self.args.keywords_loss_weight * kw_loss
        else:
            logits = self.model(mode='train', x=batch.batch_x, y=batch.batch_y)
            loss = self.loss_fn(input=logits, target=batch.batch_y)

        # Backward & Optim
        loss = loss / self.args.n_accum_batches
        loss.backward()
        self.train_steps += 1
        if self.train_steps % self.args.n_accum_batches == 0:
            self.optim.step()
            self.model.zero_grad()

        # Log
        self.writer.add_scalar('_Loss/all', loss.item(), self.train_steps)
        if self.args.use_keywords:
            self.writer.add_scalar('_Loss/response', rsp_loss.item(), self.train_steps)
            self.writer.add_scalar('_Loss/keywords', kw_loss.item(), self.train_steps)

        if self.train_steps % self.args.case_interval == 0:
            y_pred_ids = logits.argmax(dim=-1)
            y_pred = self.batch_ids_to_strings(y_pred_ids)
            if self.args.use_keywords:
                k_pred_ids = kw_logits.argmax(dim=-1)
                k_pred = self.batch_ids_to_tokens(k_pred_ids)
            else:
                k_pred = None
            self.recoder.record(mode='train', epoch=self.epoch, step=self.train_steps,
                                batch=batch, index=0, y_pred=y_pred, k_pred=k_pred)
        return loss.item()

    def train_epoch(self):
        random.shuffle(self.train_batches)
        self.model.train().zero_grad()
        pbar = tqdm(self.train_batches,
                    desc=f'[{self.args.tag}] [{self.device}] Train {self.epoch}',
                    dynamic_ncols=True)
        for batch in pbar:
            loss = self.train_batch(batch)
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        ckpt_path = os.path.join(
            self.checkpoints_dir, f'{self.args.tag}-epoch-{self.epoch}.pt')
        torch.save(self.model.state_dict(), ckpt_path)

    def fit(self):
        if self.args.train_pickle_path is not None:
            with open(self.args.train_pickle_path, 'rb') as f:
                self.train_batches = pickle.load(f)
        if self.args.test_pickle_path is not None:
            with open(self.args.test_pickle_path, 'rb') as f:
                self.test_batches = pickle.load(f)
            self.evaluator = Evaluator(batches=self.test_batches,
                                       writer=self.writer)

        if not self.args.resume:
            self.epoch = 0
            self.train_steps, self.test_steps = 0, 0
        else:
            if hasattr(self, 'train_batches'):
                self.train_steps = self.epoch * len(self.train_batches)
            if hasattr(self, 'test_batches'):
                self.test_steps = self.epoch * len(self.test_batches)

        while True:
            self.epoch += 1
            if hasattr(self, 'train_batches'):
                self.train_epoch()
            if hasattr(self, 'test_batches'):
                data = self.test_epoch()
                self.evaluator.evaluate(data, self.epoch)

    def test_epoch(self):
        self.model.eval()
        results = {}
        pbar = tqdm(self.test_batches,
                    desc=f'[{self.args.tag}] [{self.device}] Test {self.epoch}',
                    dynamic_ncols=True)
        for batch in pbar:
            result = self.test_batch(batch)
            for k, v in result.items():
                results.setdefault(k, []).extend(v)
            pbar.set_postfix({'step': self.test_steps})
        return results

    def test_batch(self, batch):
        batch.to(self.device)
        if self.args.use_keywords:
            y_pred_ids, k_pred_ids = self.model(mode='test', x=batch.batch_x)
            y_pred = self.batch_ids_to_strings(y_pred_ids)
            k_pred = self.batch_ids_to_tokens(k_pred_ids)
            result = {'y_pred': y_pred, 'k_pred': k_pred}
        else:
            y_pred_ids = self.model(mode='test', x=batch.batch_x)
            y_pred = self.batch_ids_to_strings(y_pred_ids)
            result = {'y_pred': y_pred}
            k_pred = None

        self.test_steps += 1
        if self.test_steps % self.args.case_interval == 0:
            self.recoder.record(mode='test', epoch=self.epoch, step=self.test_steps,
                                batch=batch, index=0, y_pred=y_pred, k_pred=k_pred)
        return result

    def batch_ids_to_strings(self, batch_ids):
        strings = []
        for ids in batch_ids.tolist():
            tokens = self.ids_to_tokens(ids)
            string = self.tokenizer.convert_tokens_to_string(tokens)
            strings.append(string)
        return strings

    def batch_ids_to_tokens(self, batch_ids):
        return [self.ids_to_tokens(ids) for ids in batch_ids.tolist()]

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        if tokens.count(self.tokenizer.sep_token) > 0:
            sep_pos = tokens.index(self.tokenizer.sep_token)
            tokens = tokens[:sep_pos]
        return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', '-t', required=True)
    parser.add_argument('--overfit', '-o', default=-1, type=int)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--use_keywords', action='store_true')

    parser.add_argument('--train_pickle_path')
    parser.add_argument('--test_pickle_path')
    parser.add_argument('--lr', default='1e-5', type=float)
    parser.add_argument('--response_loss_weight', default=0.5, type=float)
    parser.add_argument('--keywords_loss_weight', default=0.5, type=float)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--n_accum_batches', default=2, type=int)

    parser.add_argument('--pretrain_path', default='bert-base-uncased/pytorch_model.bin')
    parser.add_argument('--vocab_path', default='bert-base-uncased/vocab.txt')
    parser.add_argument('--checkpoints_base_dir', default='checkpoints')
    parser.add_argument('--tensorboard_base_dir', default='runs')
    parser.add_argument('--mongodb_port', default=27017, type=int)
    parser.add_argument('--mongodb_db_name', default='kwseq')
    parser.add_argument('--case_interval', default=10, type=int)

    parser.add_argument('--max_decode_len', default=30, type=int)
    args = parser.parse_args()

    trainer = Trainer(args)

    if args.overfit > 0:
        trainer.overfit(args.overfit)
    else:
        trainer.fit()

"""
python trainer.py -t overfit --overfit 200 --clear \
    --train_pickle_path=daily_train_2500.pickle

python trainer.py -t train --resume \
    --train_pickle_path=daily_train_2500.pickle \
    --test_pickle_path=daily_test_3000.pickle
"""
