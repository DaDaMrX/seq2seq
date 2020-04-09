from concurrent.futures import ThreadPoolExecutor

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sklearn


class Evaluator:

    def __init__(self, batches, writer=None):
        self.writer = writer
        self.items = []
        if hasattr(batches[0], 'batch_text_y'):
            self.y = sum((b.batch_text_y for b in batches), [])
            self.y = [[s] for s in self.y]
            self.items.append((self.y, 'y_pred', self.bleu))
        if hasattr(batches[0], 'batch_tokens_k'):
            self.k = sum((b.batch_tokens_k for b in batches), [])
            self.items.append((self.k, 'k_pred', self.kw_metric))

    def evaluate(self, data, epoch):
        futures, results = [], {}
        with ThreadPoolExecutor(max_workers=len(self.items)) as executor:
            for gt_data, key, metric_fn in self.items:
                if key in data:
                    futures.append(executor.submit(metric_fn, data[key]))
            for future in futures:
                results.update(future.result())

        if self.writer is not None:
            if 'BLEU-4' in results:
                for k in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
                    self.writer.add_scalar(f'test-BLEU/{k}', results[k], epoch)
            if 'F1' in results:
                for k in ['Accuracy', 'Precision', 'Recall', 'F1']:
                    self.writer.add_scalar(f'test-Keywords/{k}', results[k], epoch)
        return results

    def bleu(self, y_pred, ranks=[1, 2, 3, 4]):
        assert len(y_pred) == len(self.y)
        scores = {}
        for rank in ranks:
            scores[f'BLEU-{rank}'] = corpus_bleu(
                list_of_references=self.y,
                hypotheses=y_pred,
                weights=(1 / rank,) * rank,
                smoothing_function=SmoothingFunction().method1,
            )
        return scores

    def kw_metric(self, k_pred):
        assert len(k_pred) == len(self.k)
        names = ['Accuracy', 'Precision', 'Recall', 'F1']
        scores = {name: 0 for name in names}
        for k_gt, k_pred in zip(self.k, k_pred):
            truth, pred = [], []
            for word in set(k_gt + k_pred):
                truth.append(word in k_gt)
                pred.append(word in k_pred)
            scores['Accuracy'] += sklearn.metric.accuracy_score(truth, pred)
            scores['Precision'] += sklearn.metric.precision_score(truth, pred)
            scores['Recall'] += sklearn.metric.recall_score(truth, pred)
            scores['F1'] += sklearn.metric.f1_score(truth, pred)
        for name in scores:
            scores[name] /= len(self.k)
        return scores


if __name__ == '__main__':
    e = Evaluator(gt_data=[2, 3, 5, 7, 11])
    score = e.evaluate(data=[2, 3, 5, 7, 9])
    print(score)
