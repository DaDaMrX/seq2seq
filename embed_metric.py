from metric import calc_metrics, EmbeddingMetirc
import json
import os
import time
# import transformers
# hyps_path = 'results/seq2seq-tie/valid-hyps/hyps-epoch-183.txt'  # 0.878, 0.513, 0.710
# hyps_path = 'results/concat/valid-hyps/responses-epoch-200.txt'  # 0.885, 0.500, 0.712
# hyps_path = 'results/concat-kwprob-copy-cos/valid-hyps/responses-epoch-109.txt'  # 0.876, 0.516, 0.709
# hyps_path = 'results/concat-kwprob-copy-cos/valid-hyps/responses-epoch-119.txt'  # 0.878, 0.521, 0.712
#hyps_path = 'results/concat-kwprob-copy-cos/valid-hyps/responses-epoch-175.txt'  # 0.882 0.521 0.720
def cal(input,out):
    glove_path = 'metrics/glove.6B.300d.model.bin'
    print('load')
    emb = EmbeddingMetirc(glove_path)

    refs_path = f'results/{input}/valid-hyps/refs.txt'
    with open(refs_path) as f:
        refs = f.read().splitlines()
    
    dir_path = f'results/{input}/valid-hyps'
    
    num = 80
    while num < 81:
        filename = f'responses-epoch-{num}.txt'
        hyps_path = f'results/{input}/valid-hyps/'+ filename
        print(hyps_path)
        try:
            with open(hyps_path) as f:
                hyps = f.read().splitlines()
                emb_metircs = emb.embedding_metrics(hyps, refs)
                file_name = f'metric-sfs/{out}/'+ filename[:-4] + '.json'
            for k, v in emb_metircs.items():
                print(k, v)
            metrics = calc_metrics(hyps,refs,False,True,True)
            for k,v in metrics.items():
                print(k,v)
            res = {}
            res.update(metrics)
            res.update(emb_metircs)
            with open(file_name,'w') as f:
                f.write(json.dumps(res))
        except FileNotFoundError:
            print(f'no such file{hyps_path}')
        num = num + 1
if __name__ == '__main__':
    cal('all-gt','metric-all-gt')
    #cal('seq2seq-no-bert','metric-seq2seq-no-bert')
    
