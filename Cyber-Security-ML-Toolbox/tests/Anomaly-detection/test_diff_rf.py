'''
Author: your name
Date: 2021-06-21 19:53:12
LastEditTime: 2021-07-16 11:05:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/Anomaly-detection/IsolationForest.py
'''
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_and_ensemble_predict
import numpy as np
from DiFF_RF import DiFF_TreeEnsemble


if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    
    X_train,y_train,X_val,y_val,X_test,y_test,n_features,mask=get_datasets(options)
    X_train,y_train=X_train[0:300],y_train[0:300]
    X_train=X_train[y_train==0]
    y_train=y_train[y_train==0]

    # isolation_forest = IsolationForest(n_estimators=100, max_samples=256, contamination=0.1, random_state=42)
    diff_rf = DiFF_TreeEnsemble(n_trees=256) 

    diff_rf.fit(X_train)

    anomaly_scores = diff_rf.anomaly_score(X_test,alpha=0.1)
    print(y_test)
    print(anomaly_scores)
    anomaly_scores=anomaly_scores.reshape(-1,1)
    mm=MinMaxScaler()
    anomaly_scores=mm.fit_transform(anomaly_scores)
    anomaly_scores=anomaly_scores
    y_pred_=np.hstack((anomaly_scores,1-anomaly_scores))

    y_pred_=np.argmax(y_pred_, axis=1)
    print(y_pred_)


    from sklearn.metrics import roc_auc_score

            
    auc = roc_auc_score(y_test, y_pred_)
    print("AUC: {:.2%}".format (auc))

    print(classification_report(y_test,y_pred_))
