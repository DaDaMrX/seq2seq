import argparse
import logging
import os
import pickle
import random
import shutil
import time

import apex
import pynvml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

from dataset import Data
from evaluator import Evaluator
from model import KWSeq2Seq, Seq2Seq
from recoder import Recoder


def get_gpu_info():
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


class Trainer:

    def __init__(self, args):
        self.args = args
        self.logger = self.get_logger('Trainer')
        self.random = random.Random(self.args.seed)

        self.device = torch.device(self.args.device_id)
        torch.cuda.set_device(self.device)
        self.logger.info(f'Use device {self.device}')

        self.tokenizer = BertTokenizer(self.args.vocab_path)
        self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})

        self.model, self.optim = self.load_model()
        self.train_batches_all, self.test_batches_all = self.load_batches()

        if self.args.is_worker:
            self.tensorboard_dir, self.checkpoints_dir = self.prepare_dirs()
            self.writer = SummaryWriter(self.tensorboard_dir)
        self.recoder = Recoder(tag=self.args.tag,
                               clear=self.args.clear,
                               port=self.args.mongodb_port,
                               db_name=self.args.mongodb_db_name)

    def load_model(self):
        model_args = dict(
            tokenizer=self.tokenizer,
            max_decode_len=self.args.max_decode_len,
        )
        if self.args.use_keywords:
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

        self.model, self.optim = apex.amp.initialize(
            self.model, self.optim, opt_level='O2')

        if self.args.n_gpu > 1:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:29500',
                rank=self.args.rank,
                world_size=self.args.n_gpu,
            )
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.rank],
                output_device=self.args.rank,
                find_unused_parameters=True,
            )
        return self.model, self.optim

    def load_batches(self):
        self.train_batches_all = None
        self.test_batches_all = None

        if self.args.train_pickle_path:
            with open(self.args.train_pickle_path, 'rb') as f:
                batches = pickle.load(f)
            n, m = len(batches), self.args.n_gpu
            r = (n + m - 1) // m * m - n
            self.train_batches_all = batches + batches[:r]

        if self.args.test_pickle_path:
            with open(self.args.test_pickle_path, 'rb') as f:
                self.test_batches_all = pickle.load(f)

        return self.train_batches_all, self.test_batches_all

    def get_logger(self, class_name):
        colors = ['', '\033[92m', '\033[93m', '\033[94m']
        reset_color = '\033[0m'
        self.logger = logging.getLogger(__name__ + '.' + class_name)
        self.logger.setLevel(logging.INFO)
        hander = logging.StreamHandler()
        hander.setLevel(logging.INFO)
        s = f'[%(levelname)s] [{class_name}.%(funcName)s] %(message)s'
        if self.args.n_gpu > 1:
            s = f'[Rank {self.args.rank}] ' + s
            s = f' {colors[self.args.rank % 4]}' + s + reset_color
        formatter = logging.Formatter(s)
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

    def train_batch(self, batch):
        # Forward & Loss
        batch.to(self.device)
        if self.args.use_keywords:
            gt_kw_prob = 1
            rsp_logits, kw_logits = self.model(
                mode='train', x=batch.x, y=batch.y,
                k=batch.k, gt_kw_prob=gt_kw_prob)
            rsp_loss = self.loss_fn(input=rsp_logits, target=batch.y)
            kw_loss = self.loss_fn(input=kw_logits, target=batch.k)
            loss = self.args.response_loss_weight * rsp_loss + \
                self.args.keywords_loss_weight * kw_loss
        else:
            logits = self.model(mode='train', x=batch.x, y=batch.y)
            loss = self.loss_fn(input=logits, target=batch.y)

        # Backward & Optim
        loss = loss / self.args.n_accum_batches
        with apex.amp.scale_loss(loss, self.optim) as scaled_loss:
            scaled_loss.backward()
        self.train_steps += 1
        if self.train_steps % self.args.n_accum_batches == 0:
            self.optim.step()
            self.model.zero_grad()

        # Log
        if self.args.is_worker:
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
            self.recoder.record(
                mode='train', epoch=self.epoch, step=self.train_steps, rank=self.args.rank,
                texts_x=batch.texts_x[0], text_y=batch.text_y[0], y_pred=y_pred[0],
                tokens_k=batch.tokens_k[0] if 'tokens_k' in batch else None,
                k_pred=k_pred[0] if k_pred is not None else None,
            )
        return loss.item()

    def train_epoch(self):
        self.model.train()
        self.model.zero_grad()
        if self.args.is_worker:
            pbar = tqdm(self.train_batches,
                        desc=f'[{self.args.tag}] [{self.device}] Train {self.epoch}',
                        dynamic_ncols=True)
        else:
            pbar = self.train_batches
        for batch in pbar:
            loss = self.train_batch(batch)  # Core training
            if self.args.is_worker:
                pbar.set_postfix({'loss': f'{loss:.4f}'})

        if self.args.is_worker:
            ckpt_path = os.path.join(
                self.checkpoints_dir, f'{self.args.tag}-epoch-{self.epoch}.pt')
            torch.save(self.model.state_dict(), ckpt_path)

    def fit(self):
        self.epoch = 0
        self.train_steps, self.test_steps = 0, 0

        if self.test_batches_all is not None:
            self.test_batches = self.test_batches_all[self.args.rank::self.args.n_gpu]
            if self.args.is_worker:
                self.recoder.record_target(self.test_batches_all)

        while True:
            self.epoch += 1

            if self.train_batches_all is not None:
                self.random.shuffle(self.train_batches_all)
                self.train_batches = self.train_batches_all[self.args.rank::self.args.n_gpu]
                if self.args.n_gpu > 1:
                    torch.distributed.barrier()
                self.train_epoch()

            if self.test_batches_all is not None:
                if self.args.n_gpu > 1:
                    torch.distributed.barrier()
                results = self.test_epoch()
                self.recoder.record_output(results)

    def test_epoch(self):
        self.model.eval()
        results = []
        if self.args.is_worker:
            pbar = tqdm(self.test_batches,
                        desc=f'[{self.args.tag}] [{self.device}] Test {self.epoch}',
                        dynamic_ncols=True)
        else:
            pbar = self.test_batches
        for batch in pbar:
            batch_results = self.test_batch(batch)  # Core testing
            results += batch_results
            if self.args.is_worker:
                pbar.set_postfix({'step': self.test_steps})
        return results

    def test_batch(self, batch):
        batch.to(self.device)
        results = []
        if self.args.use_keywords:
            y_pred_ids, k_pred_ids = self.model(mode='test', x=batch.x)
            y_pred = self.batch_ids_to_strings(y_pred_ids)
            k_pred = self.batch_ids_to_tokens(k_pred_ids)
            for i, y, k in zip(batch.index, y_pred, k_pred):
                results.append({'epoch': self.epoch, 'index': i, 'rank': self.args.rank,
                                'y': y, 'k': k})
        else:
            y_pred_ids = self.model(mode='test', x=batch.x)
            y_pred = self.batch_ids_to_strings(y_pred_ids)
            for i, y in zip(batch.index, y_pred):
                results.append({'epoch': self.epoch, 'index': i, 'rank': self.args.rank,
                                'y': y})
            k_pred = None

        self.test_steps += 1
        if self.test_steps % self.args.case_interval == 0:
            self.recoder.record(
                mode='test', epoch=self.epoch, step=self.test_steps, rank=self.args.rank,
                texts_x=batch.texts_x[0], text_y=batch.text_y[0], y_pred=y_pred[0],
                tokens_k=batch.tokens_k[0] if 'tokens_k' in batch else None,
                k_pred=k_pred[0] if k_pred is not None else None,
            )
        return results

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


def train(rank, device_ids, args):
    args.rank = rank
    args.device_id = device_ids[rank]
    args.is_worker = rank == args.n_gpu - 1

    trainer = Trainer(args)
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', '-t', required=True)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--use_keywords', action='store_true')

    parser.add_argument('--train_pickle_path', default='')
    parser.add_argument('--test_pickle_path', default='')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--response_loss_weight', default=0.5, type=float)
    parser.add_argument('--keywords_loss_weight', default=0.5, type=float)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--n_accum_batches', default=1, type=int)

    parser.add_argument('--pretrain_path', default='pretrain/bert-base-uncased.bin')
    parser.add_argument('--vocab_path', default='pretrain/vocab.txt')
    parser.add_argument('--checkpoints_base_dir', default='checkpoints')
    parser.add_argument('--tensorboard_base_dir', default='runs')
    parser.add_argument('--mongodb_port', default=27017, type=int)
    parser.add_argument('--mongodb_db_name', default='kwseq')
    parser.add_argument('--case_interval', default=50, type=int)

    parser.add_argument('--max_decode_len', default=30, type=int)
    args = parser.parse_args()
    args.seed = random.randrange(1e10)

    assert args.n_gpu > 0
    gpu_info = get_gpu_info()[:args.n_gpu]
    device_ids = [item['id'] for item in gpu_info]

    if args.n_gpu == 1:
        train(0, device_ids, args)
    else:
        torch.multiprocessing.spawn(
            train,
            args=(device_ids, args),
            nprocs=args.n_gpu,
        )


"""
python trainer.py -t overfit --overfit 200 --clear

python trainer.py -t seq --clear --n_gpu=3 \
    --train_pickle_path=pickle/daily_train_3000.pickle \
    --test_pickle_path=pickle/daily_test_5000.pickle
"""
