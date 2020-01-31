import sklearn.metrics as metric


KW_METRIC_NAMES = ['jaccard', 'precision', 'recall', 'f1']


def cal_kw(groundtruth, prediction):       
    y_pred , y_true = [], []
    for word in set(groundtruth + prediction):
        y_pred.append(word in prediction)
        y_true.append(word in groundtruth)
    
    result = {
        'jaccard': metric.accuracy_score(y_true, y_pred),
        'precision': metric.precision_score(y_true, y_pred),
        'recall': metric.recall_score(y_true, y_pred),
        'f1': metric.f1_score(y_true, y_pred),
    }
    return result


def kw_metric(inputs, targets):
    assert len(inputs) == len(targets)
    names = ['jaccard', 'precision', 'recall', 'f1']
    metrics = {}
    for name in names:
        metrics[name] = 0
    for input, target in zip(inputs, targets):
        result = cal_kw(groundtruth=target, prediction=input)
        for name in names:
            metrics[name] += result[name]
    for name in names:
        metrics[name] /= len(inputs)
    return metrics


if __name__=='__main__':
    groundtruth = ['a', 'b', 'c', 'd','fa']
    prediction = ['e', 'a', 'd', 'k', 'm','b']
    print(cal_kw(groundtruth, prediction))
    