import time
from threading import Thread
from pprint import pprint

import pymongo
import sklearn
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


class Evaluator:

    def __init__(self, tag, host='127.0.0.1', port=27017, db_name='kwseq'):
        self.tag = tag
        self.writer = SummaryWriter(f'runs/{self.tag}')
        self.client = pymongo.MongoClient(host, port)

        self.test_collection = self.client[db_name][f'{self.tag}-test']
        self.target_y, self.target_k = [], []
        for doc in self.test_collection.find({'epoch': 0}).sort('index'):
            self.target_y.append(doc['y'])
            if 'k' in doc:
                self.target_k.append(doc['k'])
        self.len = len(self.target_y)
        assert len(self.target_k) == 0 or len(self.target_k) == self.len

        self.result_collection = self.client[db_name][f'{self.tag}-result']

    def monitor(self, start_epoch=1, period=60):
        epoch = start_epoch
        while True:
            if self.test_collection.count_documents({'epoch': epoch}) < self.len:
                time.sleep(period)
                continue
            print(f'Epoch {epoch}')
            self.evaluate(epoch)
            epoch += 1

    def evaluate(self, epoch):
        y, k = [], []
        for doc in self.test_collection.find({'epoch': epoch}).sort('index'):
            y.append(doc['y'])
            if 'k' in doc:
                k.append(doc['k'])
        assert len(y) == self.len
        assert len(k) == 0 or len(k) == self.len

        results = {'epoch': epoch}
        results.update(self.bleu(y))
        if len(k) > 0:
            results.update(self.kw_metric(k))

        pprint(results)
        self.result_collection.insert_one(results)
        if self.writer is not None:
            if 'BLEU-4' in results:
                for k in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
                    self.writer.add_scalar(f'test-BLEU/{k}', results[k], epoch)
            if 'F1' in results:
                for k in ['Accuracy', 'Precision', 'Recall', 'F1']:
                    self.writer.add_scalar(f'test-Keywords/{k}', results[k], epoch)

    def bleu(self, y_pred, ranks=[1, 2, 3, 4]):
        assert len(y_pred) == len(self.target_y)
        scores = {}
        for rank in ranks:
            scores[f'BLEU-{rank}'] = corpus_bleu(
                list_of_references=self.target_y,
                hypotheses=y_pred,
                weights=(1 / rank,) * rank,
                smoothing_function=SmoothingFunction().method1,
            )
        return scores

    def kw_metric(self, k_pred):
        assert len(k_pred) == len(self.target_k)
        names = ['Accuracy', 'Precision', 'Recall', 'F1']
        scores = {name: 0 for name in names}
        for k_gt, k_pred in zip(self.target_k, k_pred):
            truth, pred = [], []
            for word in set(k_gt + k_pred):
                truth.append(word in k_gt)
                pred.append(word in k_pred)
            scores['Accuracy'] += sklearn.metric.accuracy_score(truth, pred)
            scores['Precision'] += sklearn.metric.precision_score(truth, pred)
            scores['Recall'] += sklearn.metric.recall_score(truth, pred)
            scores['F1'] += sklearn.metric.f1_score(truth, pred)
        for name in scores:
            scores[name] /= len(self.target_k)
        return scores


if __name__ == '__main__':
    Evaluator(tag='seq').monitor()
