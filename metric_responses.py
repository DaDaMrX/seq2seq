import argparse
import os
import time

from torch.utils.tensorboard import SummaryWriter

from metric import calc_metrics, EmbeddingMetirc
from distinct_n.metrics import distinct_n_corpus_level


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', '-t', required=True)
    parser.add_argument('--epoch', '-e', type=int, required=True)
    args = parser.parse_args()

    writer = SummaryWriter(os.path.join('runs', args.tag+'-m'), flush_secs=1)

    emb = EmbeddingMetirc('metrics/glove.6B.300d.model.bin')

    refs_path = f'results/{args.tag}/test-hyps/refs.txt'
    with open(refs_path) as f:
        refs = f.read().splitlines()

    epoch = args.epoch - 1
    while True:
        epoch += 1
        hyps_path = f'results/{args.tag}/test-hyps/hyps-epoch-{epoch}.txt'
        assert os.path.exists(hyps_path)
        print(f'Load Epoch {epoch}:', hyps_path)
        with open(hyps_path) as f:
            hyps = f.read().splitlines()

        # # BLEU
        # bleu_metrics = calc_metrics(hypothesis=hyps, references=refs, 
        #                              bleu=True, rouge=False, meteor=False)
        # for k, v in bleu_metrics.items():
        #     writer.add_scalar(f'test-Bleu/{k}', v, epoch)
        #     print(f'{k}: {v:.3f}')

        # Distinct
        distinct_1 = distinct_n_corpus_level(hyps, 1)
        distinct_2 = distinct_n_corpus_level(hyps, 2)
        writer.add_scalar('Distinct/Distinct-1', distinct_1, epoch)
        writer.add_scalar('Distinct/Distinct-2', distinct_2, epoch)
        print(f'Distinct-1: {distinct_1:.3f} | Distinct-2: {distinct_2:.3f} ')

        # Embedding
        emb_metircs = emb.embedding_metrics(hyps, refs)
        for k, v in emb_metircs.items():
            writer.add_scalar(f'Embedding/{k}', v, epoch)
            print(f'{k}: {v:.3f}')

        # Rouge
        rouge_metrics = calc_metrics(hypothesis=hyps, references=refs, 
                                     bleu=False, rouge=True, meteor=False)
        for k, v in rouge_metrics.items():
            writer.add_scalar(f'Rouge/{k}', v, epoch)
            print(f'{k}: {v:.3f}')
 