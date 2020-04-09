import pymongo


class Recoder:

    def __init__(self, *, tag, clear=False, host='127.0.0.1', port=27017, db_name='kwseq'):
        self.tag = tag
        self.client = pymongo.MongoClient(host, port)
        self.collection = self.client[db_name][self.tag]
        if clear:
            self.collection.drop()

    def record(self, mode, epoch, step, batch, index, y_pred, k_pred=None):
        assert mode in ['train', 'test']
        doc = {'mode': mode, 'epoch': epoch, 'step': step}

        doc['context'] = []
        names = ['A', 'B']
        for i, sent in enumerate(batch.batch_texts_x[index]):
            doc['context'].append(names[i & 1] + ': ' + sent)
        doc['targ'] = batch.batch_text_y[index]
        doc['pred'] = y_pred[index]

        if k_pred is not None:
            doc['kw_targ'] = ' '.join(batch.batch_tokens_k[index])
            doc['kw_pred'] = ' '.join(k_pred[index])

        self.collection.insert_one(doc)


if __name__ == '__main__':
    import pickle
    from dataset import Batch

    with open('daily_test_3000.pickle', 'rb') as f:
        test_batches = pickle.load(f)
    batch = test_batches[100]

    recoder = Recoder(tag='test')
    recoder.record(
        mode='train', epoch=1, step=12,
        batch=batch, index=0,
        y_pred=batch.batch_text_y,
        k_pred=batch.batch_tokens_k,
    )
