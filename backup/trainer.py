import argparse
import pickle
import os
import shutil
import random
import threading
import json
import logging
import math

import pynvml
from pprint import pprint
import torch
import transformers
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metric import calc_metrics, EmbeddingMetirc
from kw_metric import kw_metric, KW_METRIC_NAMES
from dataset import Batch
from model import Seq2Seq, Seq2SeqKeywords


class Trainer:

    def __init__(self, args):
        self.args = args

        # Logger
        self.logger = logging.getLogger(__name__ + '.Trainer')
        self.logger.setLevel(logging.INFO)
        hander = logging.StreamHandler()
        hander.setLevel(logging.INFO)
        formatter = logging.Formatter(f'[%(levelname)s] [Tranier.%(funcName)s] %(message)s')
        hander.setFormatter(formatter)
        self.logger.addHandler(hander)

        # Tokenizer
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args.pretrain_dir)
        self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})

        # Devices
        if args.n_devices <= 0:
            self.device = torch.device('cpu')
        else:
            device_ids, info = self.get_devices(args.n_devices)
            self.device = torch.device(device_ids[0])
            for item in info:
                self.logger.info(f'Use GPU {item["index"]} (free: {item["free"]} MB)')

        # Datasets
        with open(args.train_pickle_path, 'rb') as f:
            self.train_batches = pickle.load(f)
        with open(args.valid_pickle_path, 'rb') as f:
            self.valid_batches = pickle.load(f)
        for batch in tqdm(self.train_batches, desc='Train Batches'):
            batch.batch_x = batch.batch_x.to(self.device)
            batch.batch_y = batch.batch_y.to(self.device)
            batch.batch_k = batch.batch_k.to(self.device)
        for batch in tqdm(self.valid_batches, desc='Valid Batches'):
            batch.batch_x = batch.batch_x.to(self.device)
            batch.batch_k = batch.batch_k.to(self.device)

        # Model and optimizer
        if not args.no_keywords:
            self.model = Seq2SeqKeywords(
                pretrain_dir=args.pretrain_dir,
                tokenizer=self.tokenizer,
            )
        else:
            self.model = Seq2Seq(
                pretrain_dir=args.pretrain_dir,
                tokenizer=self.tokenizer,
            )
        self.logger.info(f'Moving model to {self.device}...')
        self.model = self.model.to(self.device)
        self.logger.info(f'Moving model to {self.device} done.')
        if args.n_devices > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        # Optim
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        # for group in self.optim.param_groups:
        #     group['initial_lr'] = group['lr']
        # num_warmup_epoch = 2 * (len(self.train_batches) // self.args.n_accum_batches)
        # num_training_epoch = 300 * (len(self.train_batches) // self.args.n_accum_batches)
        # def lr_lambda(current_epoch):
        #     if current_epoch < num_warmup_epoch:
        #         return float(current_epoch) / float(max(1, num_warmup_epoch))
        #     progress = float(current_epoch - num_warmup_epoch) / float(max(1, num_training_epoch - num_warmup_epoch))
        #     return max(0., 0.5 * (1. + math.cos(math.pi * progress)))
        # last_epoch = 198 * (len(self.train_batches) // self.args.n_accum_batches)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda, last_epoch=last_epoch)

        # Dirs
        self.prepare_dirs()
        self.writer = SummaryWriter(self.tensorboard_dir, flush_secs=1)

        # Others
        self.tag = self.args.tag
        self.has_keywords = not args.no_keywords
        self.n_accum_batches = args.n_accum_batches
        self.response_loss_lambda = args.response_loss_lambda
        self.keywords_loss_lambda = args.keywords_loss_lambda
        self.case_interval = args.case_interval
        self.max_decode_len = args.max_decode_len

    def prepare_dirs(self):
        # Base dir
        for dir_path in [self.args.tensorboard_base_dir,
                         self.args.checkpoints_base_dir,
                         self.args.results_base_dir]:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        
        # Base tag dir
        self.tensorboard_dir = os.path.join(self.args.tensorboard_base_dir, self.args.tag)
        self.checkpoints_dir = os.path.join(self.args.checkpoints_base_dir, self.args.tag)
        self.results_dir = os.path.join(self.args.results_base_dir, self.args.tag)
        for dir_path in [self.tensorboard_dir,
                         self.checkpoints_dir,
                         self.results_dir]:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

        # Results train dir
        self.train_texts_dir = os.path.join(self.results_dir, 'train-texts')
        if not os.path.exists(self.train_texts_dir):
                os.mkdir(self.train_texts_dir)

        # Results valid dir
        if hasattr(self, 'valid_batches'):
            self.valid_hyps_dir = os.path.join(self.results_dir, 'valid-hyps')
            self.valid_texts_dir = os.path.join(self.results_dir, 'valid-texts')
            self.valid_kw_dir = os.path.join(self.results_dir, 'valid-keywords')
            for dir_path in [self.valid_hyps_dir, self.valid_texts_dir, self.valid_kw_dir]:
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)

        # # Results tag test dir
        # if hasattr(self, 'test_batches'):
        #     self.test_hyps_dir = os.path.join(self.results_dir, 'test-hyps')
        #     self.test_texts_dir = os.path.join(self.results_dir, 'test-texts')
        #     self.test_kw_dir = os.path.join(self.results_dir, 'test-kw')
        #     if not os.path.exists(self.test_hyps_dir):
        #         os.mkdir(self.test_hyps_dir)
        #     if not os.path.exists(self.test_texts_dir):
        #         os.mkdir(self.test_texts_dir)
        #     if not os.path.exists(self.test_kw_dir):
        #         os.mkdir(self.test_kw_dir)
    
    def get_devices(self, num):
        pynvml.nvmlInit()
        n_devices = pynvml.nvmlDeviceGetCount()
        mb = 1024 * 1024
        info = []
        for index in range(n_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info.append({
                'index': index,
                'free': memory.free // mb,
                'used': memory.used // mb,
                'total': memory.total // mb,
                'ratio': memory.used / memory.total,
            })
        info.sort(key=lambda x: x['free'], reverse=True)
        info = info[:num]
        device_ids = [x['index'] for x in info]
        return device_ids, info

    def loss_fn(self, input, target):
        loss = torch.nn.functional.cross_entropy(
            input=input.reshape(-1, input.size(-1)),
            target=target[:, 1:].reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction='mean',
        )
        return loss

    def ids_to_string(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        if tokens.count(self.tokenizer.sep_token) > 0:
            sep_pos = tokens.index(self.tokenizer.sep_token)
            tokens = tokens[:sep_pos]
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        if tokens.count(self.tokenizer.sep_token) > 0:
            sep_pos = tokens.index(self.tokenizer.sep_token)
            tokens = tokens[:sep_pos]
        return tokens

    def get_case_text(self, batch, index, pred_ids, pred_kw_ids=None):
        speaker_dict = {True: 'A', False: 'B'}
        speaker = True
        speaker_posts = []
        for post in batch.batch_texts_x[index]:
            speaker_posts.append(speaker_dict[speaker] + ': ' + post.strip())
            speaker = not speaker
        post = '\n'.join(speaker_posts)

        targ = 'Targ-' + speaker_dict[speaker] + ': ' + batch.batch_text_y[index]

        pred = self.ids_to_string(pred_ids[index].tolist())
        pred = 'Pred-' + speaker_dict[speaker] + ': ' + pred

        text = f'{post}\n{targ}\n{pred}'

        if pred_kw_ids is not None:
            targ_kw_text = 'Targ-KW: ' + ' '.join(batch.batch_tokens_k[index])

            tokens = self.tokenizer.convert_ids_to_tokens(pred_kw_ids[index].tolist())
            if tokens[0] == self.tokenizer.bos_token:
                tokens = tokens[1:]
            if tokens.count(self.tokenizer.sep_token) > 0:
                sep_pos = tokens.index(self.tokenizer.sep_token)
                tokens = tokens[:sep_pos]
            pred_kw_text = 'Pred-KW: ' + ' '.join(tokens)

            text += f'\n{targ_kw_text}\n{pred_kw_text}'

        return text
        # return f'Epoch {self.epoch}\n' + text

    def train_batch(self, batch, batch_idx):
        if self.has_keywords:
            # Forward & loss
            warm = 50
            total = 200
            if self.epoch < warm:
                gt_kw_prob = 1.0
            elif self.epoch <= total:
                progress = (self.epoch - warm) / (total - warm)
                gt_kw_prob = 0.5 * (1 + math.cos(math.pi * progress))
                gt_kw_prob = max(0, gt_kw_prob)
            else:
                gt_kw_prob = 0
            self.writer.add_scalar('lr/kw_prob', gt_kw_prob, self.epoch)
            # if self.epoch <= 80:
            #     gt_kw_prob = 1.0
            # elif self.epoch >= 83:
            #     gt_kw_prob = 0
            # else:
            #     gt_kw_prob = 82 - self.epoch
            logits, kw_logits = self.model(mode='train', x=batch.batch_x, 
                y=batch.batch_y, k=batch.batch_k, gt_kw_prob=gt_kw_prob)
            rsp_loss = self.loss_fn(input=logits, target=batch.batch_y)
            kw_loss = self.loss_fn(input=kw_logits, target=batch.batch_k)
            loss = self.response_loss_lambda * rsp_loss + self.keywords_loss_lambda * kw_loss

            # Gradient
            loss = loss / self.n_accum_batches
            loss.backward()
            if (batch_idx + 1) % self.n_accum_batches == 0:
                self.optim.step()
                # self.scheduler.step()
                self.model.zero_grad()
        else:
            logits = self.model(mode='train', x=batch.batch_x, y=batch.batch_y)
            loss = self.loss_fn(input=logits, target=batch.batch_y)
            loss = loss / self.n_accum_batches
            loss.backward()
            if (batch_idx + 1) % self.n_accum_batches == 0:
                self.optim.step()
                self.model.zero_grad()

        # Show loss
        self.train_steps += 1
        self.writer.add_scalar('_loss/total', loss.item(), self.train_steps)
        if self.has_keywords:
            self.writer.add_scalar('_loss/response', rsp_loss.item(), self.train_steps)
            self.writer.add_scalar('_loss/keywords', kw_loss.item(), self.train_steps)
        # if (batch_idx + 1) % self.n_accum_batches == 0:
        #     self.writer.add_scalar('lr', self.scheduler.get_lr()[0], self.train_steps)

        # Show text
        if self.train_steps % self.case_interval == 0:
            pred_ids = logits.argmax(dim=-1)
            pred_kw_ids = kw_logits.argmax(dim=-1) if self.has_keywords else None
            text = self.get_case_text(batch, 0, pred_ids, pred_kw_ids)
            text = f'Step: {self.train_steps}\n' + text
            case_path = os.path.join(self.train_texts_dir, f'train-epoch-{self.epoch}.txt')
            with open(case_path, 'a') as f:
                f.write('\n\n' + text) 

        return loss.item()
 
    def overfit(self):
        # self.logger.info('Overfit')
        # self.train_steps = 0
        # self.valid_steps = 0
        # self.epoch = 0
        # self.train_batches = self.train_batches[10:12]
        # self.valid_batches = self.valid_batches[5:7]
        # while True:
        #     self.epoch += 1
        #     self.train_epoch()
        #     pred_responses, pred_keywords = self.valid_epoch()
        #     self.valid_metric(pred_responses, pred_keywords)

        self.model.train()
        batch = self.train_batches[100]
        self.epoch = 0

        # Train
        pbar = tqdm(range(300), desc='Overfit', dynamic_ncols=True)
        self.train_steps = 0
        for i in pbar:
            loss = self.train_batch(batch, i)
            pbar.set_postfix({'loss': f'{loss:.5f}'})

        # Valid
        self.valid_steps = 0
        hyps, pred_kws = self.valid_batch(batch)
        metrics = calc_metrics(hypothesis=hyps, references=batch.batch_text_y,
                               bleu=True, rouge=False, meteor=False)
        pprint(metrics)
        metrics = kw_metric(inputs=pred_kws, targets=batch.batch_tokens_k)
        pprint(metrics)

    def train_epoch(self):
        # Train mode & shuffle
        self.model.train()
        random.shuffle(self.train_batches)
        
        # Iter batch
        self.model.zero_grad()
        pbar = tqdm(self.train_batches,
                    desc=f'[{self.tag}] Train Epoch {self.epoch}', 
                    dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            loss = self.train_batch(batch, batch_idx)
            pbar.set_postfix({'step': self.train_steps, 'loss': f'{loss:.5f}'})

        # Save checkpoint
        ckpt_path = os.path.join(
            self.checkpoints_dir, f'model-{self.tag}-epoch-{self.epoch}.pt')
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), ckpt_path)
        else:
            torch.save(self.model.state_dict(), ckpt_path)

    def fit(self):
        self.train_steps = 0
        self.valid_steps = 0
        self.epoch = 0
        while True:
            self.epoch += 1
            if self.train_batches is not None:
                self.train_epoch()
            if self.valid_batches is not None and \
                (self.epoch <= 5 or self.epoch >= 100 or self.epoch % 10 == 0):
                pred_responses, pred_keywords = self.valid_epoch()
                self.valid_metric(pred_responses, pred_keywords)

    def continue_fit(self):
        # Find latest checkpoint
        latest_epoch, latest_path = -1, None
        for name in os.listdir(self.checkpoints_dir):
            epoch = int(os.path.splitext(name)[0].split('-')[-1])
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = name
        latest_path = os.path.join(self.checkpoints_dir, latest_path)

        # latest_epoch = 198
        # latest_path = 'checkpoints/concat-kwprob-copy-cos/model-concat-kwprob-copy-cos-epoch-198.pt'

        # Load latest checkpoint
        self.logger.info(f'Load checkpoint from {latest_path}')
        state_dict = torch.load(latest_path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state_dict, strict=True)
        else:
            self.model.load_state_dict(state_dict, strict=True)
        
        self.epoch = latest_epoch
        self.train_steps = self.epoch * len(self.train_batches)
        if self.valid_batches is not None:
            self.valid_steps = self.epoch * len(self.valid_batches)

        while True:
            self.epoch += 1
            if self.train_batches is not None:
                self.train_epoch()
            if self.valid_batches is not None and \
                (self.epoch <= 5 or self.epoch >= 100):
                pred_responses, pred_keywords = self.valid_epoch()
                self.valid_metric(pred_responses, pred_keywords)

    def load_and_metric(self, start_epoch, glove_path=None):
        self.logger.info(f'Load and metric from epoch {start_epoch}')
        if glove_path:
            self.logger.info('Load glove embedding...')
            emb = EmbeddingMetirc(glove_path)
            self.logger.info('Load glove embedding done.')
        else:
            emb = None

        self.epoch = start_epoch
        if self.valid_batches is not None:
            self.valid_steps = (self.epoch - 1) * len(self.valid_batches)
        while True:
            ckpt_path = os.path.join('checkpoints/kw-sfs/', f'model-kw-sfs-epoch-{self.epoch}.pt')
            print(ckpt_path)
            assert os.path.exists(ckpt_path)
            self.logger.info(f'Load checkpoint from "{ckpt_path}"')
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            
            if self.valid_batches is not None:
                pred_responses, pred_keywords = self.valid_epoch()
                self.valid_metric(pred_responses, pred_keywords)
            self.epoch += 1

    def valid_epoch(self):
        self.model.eval()

        hyps, kw_tokens = [], []
        pbar = tqdm(self.valid_batches, 
                    desc=f'[{self.tag}] Valid Epoch {self.epoch}', 
                    dynamic_ncols=True)
        for batch in pbar:
            if self.has_keywords:
                batch_hyps = self.valid_batch(batch)
                hyps.extend(batch_hyps)
                #batch_hyps, batch_kw_tokens = self.valid_batch(batch)
                #hyps.extend(batch_hyps)
                #kw_tokens.extend(batch_kw_tokens)
            else:
                batch_hyps = self.valid_batch(batch)
                hyps.extend(batch_hyps)
            pbar.set_postfix({'step': self.valid_steps})
        
        return hyps, kw_tokens
        

    def valid_batch(self, batch):
        with torch.no_grad():
            if self.has_keywords:
                y_pred = self.model(mode='valid', x=batch.batch_x, k=batch.batch_k, max_len=self.max_decode_len)
            else:
                y_pred = self.model(mode='valid', x=batch.batch_x, max_len=self.max_decode_len)
                #kw_pred = None

        self.valid_steps += 1

        #text = self.get_case_text(batch, 0, y_pred, kw_pred)
        case_path = os.path.join(self.valid_texts_dir, f'valid-epoch-{self.epoch}.txt')
        #with open(case_path, 'a') as f:
            #f.write('\n\n' + text)

        hyps = [self.ids_to_string(hyp) for hyp in y_pred.tolist()]
        #if self.has_keywords:
        #   kw_tokens = [self.ids_to_tokens(kw_ids) for kw_ids in kw_pred.tolist()]
        #if self.has_keywords:
        #   return hyps, kw_tokens
        #else:
        return hyps

    def valid_metric(self, pred_responses, pred_keywords):
        # save valid-responses/refs.txt
        if not hasattr(self, 'response_refs'):
            self.response_refs = []
            for batch in tqdm(self.valid_batches, desc='Ref'):
                self.response_refs.extend(batch.batch_text_y)
            with open(os.path.join(self.valid_hyps_dir, f'refs.txt'), 'w') as f:
                f.write('\n'.join(self.response_refs))

        # metric
        threading.Thread(
            target=self.metric_thread,
            args=(pred_responses, self.response_refs, self.epoch, None),
        ).start()

        if self.has_keywords:
            # save valid-keywords/refs-rank0.txt
            if not hasattr(self, 'tgt_kw_tokens'):
                self.tgt_kw_tokens = []
                for batch in self.valid_batches:
                    self.tgt_kw_tokens.extend(batch.batch_tokens_k)
                kw_refs = [' '.join(kws) for kws in self.tgt_kw_tokens]
                with open(os.path.join(self.valid_kw_dir, f'refs.txt'), 'w') as f:
                    f.write('\n'.join(kw_refs))
                
            threading.Thread(
                target=self.kw_metric_thread, 
                args=(pred_keywords, self.tgt_kw_tokens, self.epoch),
            ).start()

    def metric_thread(self, hyps, refs, epoch, emb):
        hyps_path = os.path.join(self.valid_hyps_dir, f'responses-epoch-{self.epoch}.txt')
        with open(hyps_path, 'w') as f:
            f.write('\n'.join(hyps))

        metrics = calc_metrics(hypothesis=hyps, references=refs, bleu=True, rouge=False, meteor=False)

        self.writer.add_scalar(f'test-Bleu/Bleu_1', metrics['Bleu_1'], epoch)
        self.writer.add_scalar(f'test-Bleu/Bleu_2', metrics['Bleu_2'], epoch)
        self.writer.add_scalar(f'test-Bleu/Bleu_3', metrics['Bleu_3'], epoch)
        self.writer.add_scalar(f'test-Bleu/Bleu_4', metrics['Bleu_4'], epoch)
        # self.writer.add_scalar(f'{mode}-Rouge/Rouge-1', metrics['rouge-1'], self.epoch)
        # self.writer.add_scalar(f'{mode}-Rouge/Rouge-2', metrics['rouge-2'], self.epoch)
        # self.writer.add_scalar(f'{mode}-Rouge/Rouge-l', metrics['rouge-l'], self.epoch)
        # self.writer.add_scalar(f'{mode}-Meteor', metrics['METEOR'], self.epoch)

        metrics['epoch'] = epoch
        metrics_path = os.path.join(self.results_dir, f'metrics.json')
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        if emb is not None:
            emb_metircs = emb.embedding_metrics(hyps, refs)
            self.writer.add_scalar(f'Embedding/Average', emb_metircs["Average"], epoch)
            self.writer.add_scalar(f'Embedding/Extrema', emb_metircs["Extrema"], epoch)
            self.writer.add_scalar(f'Embedding/Greedy', emb_metircs["Greedy"], epoch)

            emb_metircs["epoch"] = epoch
            results_path = os.path.join(self.results_dir, f'embed.txt')
            with open(results_path, 'a') as f:
                f.write(json.dumps(emb_metircs) + '\n')

    def kw_metric_thread(self, preds, targs, epoch):
        keywords = [' '.join(kws) for kws in preds]
        kw_path = os.path.join(self.valid_kw_dir, f'keywords-epoch-{self.epoch}.txt')
        with open(kw_path, 'w') as f:
            f.write('\n'.join(keywords))

        metrics = kw_metric(inputs=preds, targets=targs)
        for name in KW_METRIC_NAMES:
            self.writer.add_scalar(f'test-Keywords/{name}', metrics[name], epoch)

        metrics['epoch'] = self.epoch
        metrics_path = os.path.join(self.results_dir, f'valid-kw-metrics.json')
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', '-t', required=True)
    parser.add_argument('--continues', '-c', action='store_true')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--metric_from', default=0, type=int)

    parser.add_argument('--train_pickle_path', default='data/dialogues_train_keywords_2500_texty.pickle')
    parser.add_argument('--valid_pickle_path', default='data/dialogues_test_keywords_2500_texty.pickle')
    parser.add_argument('--n_devices', default=1, type=int)

    parser.add_argument('--pretrain_dir', default='bert-base-uncased/')
    parser.add_argument('--checkpoints_base_dir', default='checkpoints')
    parser.add_argument('--results_base_dir', default='results')
    parser.add_argument('--tensorboard_base_dir', default='runs')
    parser.add_argument('--case_interval', default=10, type=int)
    parser.add_argument('--max_decode_len', default=30, type=int)
    parser.add_argument('--n_accum_batches', default=2, type=int)
    parser.add_argument('--response_loss_lambda', default=0.5, type=float)
    parser.add_argument('--keywords_loss_lambda', default=0.5, type=float)

    parser.add_argument('--lr', default='1e-5', type=float)
    parser.add_argument('--no_keywords', action='store_true')
    args = parser.parse_args()

    trainer = Trainer(args)
    
    if args.overfit > 0:
        trainer.overfit()
    elif args.metric_from > 0:
        trainer.load_and_metric(start_epoch=args.metric_from)
    elif args.continues:
        trainer.continue_fit()
    else:
        trainer.fit()
