
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
import numpy as np
from csmt.utils import get_number_labels

def get_class_scores(labels, predictions):
    len_labels=get_number_labels(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    if len_labels<=2:
        f1 = metrics.f1_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        # fpr,tpr,trhresholds = roc_curve(y_true=labels,y_score=predictions)
        # roc_auc = auc(x=fpr,y=tpr)
        roc_auc=np.round(roc_auc_score(labels, predictions), decimals=6)
        asr=(1-metrics.recall_score(labels, predictions))*100
        return accuracy, f1, precision, recall,roc_auc,asr
    else:
        f1 = metrics.f1_score(labels, predictions,average='micro')
        precision = metrics.precision_score(labels, predictions,average='micro')
        recall = metrics.recall_score(labels, predictions,average='micro')
        roc_auc=0
        asr=(1-recall)*100
        return accuracy, f1, precision, recall,roc_auc,asr
    

def print_results(labels, predictions):
    headers = ['accuracy', 'f1', 'precision', 'recall', 'detection_rate']
    rows=[]
    result=get_class_scores(labels, predictions)
    row=list(result)
    rows.append(row)
    print(tabulate(rows, headers=headers))
