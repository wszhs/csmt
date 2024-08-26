'''
Author: your name
Date: 2021-07-20 15:23:44
LastEditTime: 2021-07-28 09:50:32
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_classifer.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_predict_anomaly
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from sklearn.preprocessing import MinMaxScaler
from csmt.figure import CFigure
from math import ceil
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)
#torch.set_default_tensor_type(torch.DoubleTensor)

def plot_anomaly(y_test,y_pred):
    y_pred_=y_pred[0][:,1]
    anomaly_scores=y_pred_[y_test==1]
    normal_scores=y_pred_[y_test==0]
    #绘制异常检测异常值图
    plt.figure(figsize=(15, 10))
    plt.hist(normal_scores, bins=100, color='blue')
    plt.hist(anomaly_scores, bins=100, color='orange')
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Number of Data Points', fontsize=14)
    plt.show()

def plot_anomaly_vec(X_test,y_test,y_pred):
    thres=0
    scores_vec=y_pred[0][:,1]
    X_test_1=X_test[y_test==1]
    abnormaly_scores_vec=scores_vec[y_test==1]
    X_test_0=X_test[y_test==0]
    normaly_scores_vec=scores_vec[y_test==0]
    plt.figure(figsize=(15, 10))
    plt.scatter(np.linspace(0,len(X_test_1)-1,len(X_test_1)),abnormaly_scores_vec,s=5,c='orange')
    plt.scatter(np.linspace(0,len(X_test_0)-1,len(X_test_0)),normaly_scores_vec,s=5,c='blue')
    # plt.plot(np.linspace(0,len(X_test)-1,len(X_test)),[thres]*len(X_test),c='black')
    plt.show()

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    
    X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)

    trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)
    
     # trained_models=models_load(datasets_name,orig_models_name)
    y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)
    
    plot_anomaly(y_test,y_pred)
    # plot_anomaly_vec(X_test,y_test,y_pred)

    table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')









