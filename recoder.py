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

    def record(self, mode, epoch, step, batch, index, y_pred, k_pred=None):
        assert mode in ['train', 'test']
        doc = {'mode': mode, 'epoch': epoch, 'step': step}

        doc['context'] = []
        names = ['A', 'B']
        for i, sent in enumerate(batch.texts_x[index]):
            doc['context'].append(names[i & 1] + ': ' + sent)
        doc['targ'] = batch.text_y[index]
        doc['pred'] = y_pred[index]

        if k_pred is not None:
            doc['kw_targ'] = ' '.join(batch.tokens_k[index])
            doc['kw_pred'] = ' '.join(k_pred[index])

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
