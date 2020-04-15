import argparse
import os
import time
from pprint import pprint

import pymongo
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.tensorboard import SummaryWriter


class Evaluator:

    def __init__(self, args):
        self.args = args
        if self.args.tb_tag:
            tb_dir = os.path.join(self.args.tensorboard_base_dir, self.args.tb_tag)
            self.writer = SummaryWriter(tb_dir)

        self.client = pymongo.MongoClient(self.args.mongodb_uri)
        self.db = self.client[self.args.mongodb_db]
        self.test_collection = self.db[f'{self.args.tag}-test']
        self.result_collection = self.db[f'{self.args.tag}-result']

        self.target_y, self.target_k = self.load_target()
        self.len = len(self.target_y)
        print((f'Find {len(self.target_y)} Target repsones, '
               f'{len(self.target_k)} Target keywords.'))

    def load_target(self):
        while self.test_collection.count_documents({'epoch': 0}) == 0:
            time.sleep(self.args.period)

        self.target_y, self.target_k = [], []
        for doc in self.test_collection.find({'epoch': 0}).sort('index'):
            self.target_y.append(doc['y'])
            if self.args.use_keywords:
                self.target_k.append(doc['k'])

        return self.target_y, self.target_k

    def monitor(self):
        epoch = self.args.start_epoch
        while True:
            print(f'Waiting Epoch {epoch}...')
            while self.test_collection.count_documents({'epoch': epoch}) < self.len:
                time.sleep(self.args.period)
            print(f'Find Epoch {epoch}:')
            self.evaluate(epoch)
            epoch += 1

    def evaluate(self, epoch):
        y, k = [], []
        for doc in self.test_collection.find({'epoch': epoch}).sort('index'):
            y.append(doc['y'])
            if self.args.use_keywords:
                k.append(doc['k'])
        print(f'Find {len(y)} Repsones, {len(k)} Keywords.')

        results = {'epoch': epoch}
        results.update(self.distinct(y))
        results.update(self.bleu(y))
        if self.args.use_keywords:
            results.update(self.kw_metric(k))

        pprint(results)
        self.result_collection.insert_one(results)
        if hasattr(self, 'writer'):
            if 'BLEU-4' in results:
                for k in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
                    self.writer.add_scalar(f'test-BLEU/{k}', results[k], epoch)
            if 'Distinct-1' in results:
                for k in ['Distinct-1', 'Distinct-2']:
                    self.writer.add_scalar(f'test-Distinct/{k}', results[k], epoch)
            if 'F1' in results:
                for k in ['Accuracy', 'Precision', 'Recall', 'F1']:
                    self.writer.add_scalar(f'test-Keywords/{k}', results[k], epoch)

    def bleu(self, y_pred, ranks=[1, 2, 3, 4]):
        assert len(y_pred) == len(self.target_y)
        target_y = [[s] for s in self.target_y]
        scores = {}
        for rank in ranks:
            scores[f'BLEU-{rank}'] = corpus_bleu(
                list_of_references=target_y,
                hypotheses=y_pred,
                weights=(1 / rank,) * rank,
                smoothing_function=SmoothingFunction().method1,
            )
        return scores

    def distinct(self, y_pred, ranks=[1, 2]):

        def ngrams(seq, n):
            seq = tuple(seq)
            return [seq[l:l + n] for l in range(len(seq) - n + 1)]

        def distint(seq, n):
            if len(seq) == 0:
                return 0.0
            return len(set(ngrams(seq, n))) / len(seq)

        scores = {}
        for rank in ranks:
            s = sum(distint(sent.split(), rank) for sent in y_pred)
            scores[f'Distinct-{rank}'] = s / len(y_pred)
        return scores

    def kw_metric(self, k_pred):
        assert len(k_pred) == len(self.target_k)
        names = ['Accuracy', 'Precision', 'Recall', 'F1']
        scores = {name: 0 for name in names}
        for k_gt, k in zip(self.target_k, k_pred):
            truth, pred = [], []
            for word in set(k_gt + k):
                truth.append(word in k_gt)
                pred.append(word in k)
            scores['Accuracy'] += accuracy_score(truth, pred)
            scores['Precision'] += precision_score(truth, pred)
            scores['Recall'] += recall_score(truth, pred)
            scores['F1'] += f1_score(truth, pred)
        for name in scores:
            scores[name] /= len(self.target_k)
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', '-t', required=True)
    parser.add_argument('--tb_tag', default=None)
    parser.add_argument('--use_keywords', '-k', action='store_true')
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--period', default=60, type=int)
    parser.add_argument('--tensorboard_base_dir', default='runs')
    parser.add_argument('--mongodb_uri', default='mongodb://root:mongodbv100@localhost:27017')
    parser.add_argument('--mongodb_db', default='kwseq')
    args = parser.parse_args()

    Evaluator(args).monitor()


"""
python evaluator.py -t seq --tb_tag seq-metric
"""
