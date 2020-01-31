from sklearn.metrics import recall_score
from kw_metric import cal_kw
import os
import json
def cal_recall(gt,pred):
    y_true = []
    y_pred = []
    for k in set(gt+pred):
        y_true.append(k in gt)
        y_pred.append(k in pred)
    return recall_score(y_true,y_pred)
    
def cal(tag,epoch):
    kw_filename = f'valid-keywords/keywords-epoch-{epoch}.txt'
    hys_filename = f'valid-hyps/responses-epoch-{epoch}.txt'
    dir_name = f'results/{tag}'
    gts = []
    preds = []
    print(os.path.join(dir_name,kw_filename))
    i=0
    with open(os.path.join(dir_name,kw_filename)) as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split()) != 0:
                gts.append(line.split())
            else:
                print(i)
                gts.append(line.split())
            i+=1
    with open(os.path.join(dir_name,hys_filename)) as f:
        lines = f.readlines()
        for line in lines:
            preds.append(line.split())
           
    res = 0 
    for gt,pred in zip(gts,preds):
        if len(gt)!=0:
            res += cal_recall(gt,pred)
    print(len(gts))
    res /= len(gts)
    dir_out = f'metric-sfs/recall/{tag}'
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    with open(os.path.join(dir_out,f'{epoch}.txt'),'w') as f:
        f.write(str(res))
    return res

if __name__ == '__main__':
    '''
    groundtruth = ['a', 'b', 'c', 'd','fa']
    prediction = ['e', 'a', 'd', 'k', 'm','b']
    print(cal_kw(groundtruth, prediction))
    print(cal_recall(groundtruth, prediction))
    '''
    #res = cal('kw-no-bert','195')
    
    #res = cal('kw-sfs','289')
    
    #res = cal('all-gt','111')
    
    #res = cal('kw-02','270')
    res = cal('all-gt','80')
    print(res)


