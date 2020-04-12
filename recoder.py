import pymongo


class Recoder:

    def __init__(self, tag, clear=False, host='127.0.0.1', port=27017, db_name='kwseq'):
        self.tag = tag
        self.client = pymongo.MongoClient(host, port)
        self.log_collection = self.client[db_name][f'{self.tag}-log']
        self.test_collection = self.client[db_name][f'{self.tag}-test']
        if clear:
            self.log_collection.drop()
            self.test_collection.drop()

    def record(self, mode, epoch, step, rank, texts_x,
               text_y, y_pred, tokens_k=None, k_pred=None):
        assert mode in ['train', 'test']
        doc = {
            'mode': mode, 'epoch': epoch,
            'step': step, 'rank': rank,
        }

        doc['context'] = []
        names = ['A', 'B']
        for i, sent in enumerate(texts_x):
            doc['context'].append(names[i & 1] + ': ' + sent)
        doc['targ'] = text_y
        doc['pred'] = y_pred

        if k_pred is not None:
            doc['kw_targ'] = ' '.join(tokens_k)
            doc['kw_pred'] = ' '.join(k_pred)

        self.log_collection.insert_one(doc)

    def record_target(self, batches):
        docs = []
        for batch in batches:
            for i in range(len(batch.index)):
                d = {
                    'epoch': 0,
                    'index': batch.index[i],
                    'y': batch.text_y[i],
                }
                if 'tokens_k' in batch:
                    d['k'] = batch.tokens_k[i]
                docs.append(d)
        self.test_collection.insert_many(docs)

    def record_output(self, output):
        self.test_collection.insert_many(output)


if __name__ == '__main__':
    import pickle
    from dataset import Data

    with open('daily_test_3000.pickle', 'rb') as f:
        test_batches = pickle.load(f)
    batch = test_batches[100]

    recoder = Recoder(tag='test')
    recoder.record(
        mode='train', epoch=1, step=12,
        batch=batch, index=0,
        y_pred=batch.text_y,
        k_pred=batch.tokens_k,
    )
