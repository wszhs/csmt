import sys

from pandas.core import base
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import csmt.Interpretability.sage as sage
import os

import matplotlib.pyplot as plt
from csmt.classifiers.scores import get_class_scores
import copy
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,get_raw_datasets
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest,VarianceThreshold,mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from minepy import MINE
from csmt.active_learning.alipy import ToolBox
from csmt.active_learning.alipy.experiment import ExperimentAnalyser
from sklearn import metrics
import torch
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)

# 计算欧氏距离
def distEclud(vecA, vecB):
    vecA=np.array(vecA)
    vecB=np.array(vecB)
    return np.sqrt(np.sum(np.power((vecA - vecB), 2)))

def get_distribute(x):
    len_distribute=len(x)
    x_all=0
    for i in range(len_distribute):
        x_all=x_all+x[i]
    distribute=[]
    for i in range(len_distribute):
        distribute.append(x[i]/x_all)
    return distribute

def feature_selection(X,y):
    # 互信息
    # MI=mutual_info_classif(X,y)
    # print(MI)
    # MIC
    n=30
    def mic(x,y):
        m = MINE()
        m.compute_score(x,y)
        return (m.mic(),0.5)
    X_selection = SelectKBest(lambda X, Y: tuple(map(tuple,np.array(list(map(lambda x:mic(x, Y), X.T))).T)),k=n).fit_transform(X,y)
    return X_selection,y

def data_selection(X,y):
    
    alibox = ToolBox(X=X, y=y, query_type='AllLabels')
    # Split data
    alibox.split_AL(test_ratio=0.3, initial_label_rate=0.01, split_count=10)
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 100)
    
    strategy_name_arr=['QueryInstanceUncertainty']
    # strategy_name_arr=['QueryInstanceRandom','QueryInstanceUncertainty','QueryInstanceGraphDensity','QueryInstanceCoresetGreedy']
    def strategy_query(strategy_name):
        _result = []
        for round in range(2):
            # Get the data split of one fold experiment
            train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
            if strategy_name in ('QueryInstanceGraphDensity','QueryInstanceCoresetGreedy'):
                _Strategy = alibox.get_query_strategy(strategy_name=strategy_name,train_idx=train_idx)
            else:
                _Strategy = alibox.get_query_strategy(strategy_name=strategy_name)
            # Get intermediate results saver for one fold experiment
            saver = alibox.get_stateio(round)

            while not stopping_criterion.is_stop():
                # Select a subset of Uind according to the query strategy
                # Passing model=None to use the default model for evaluating the committees' disagreement
                select_ind = _Strategy.select(label_ind, unlab_ind, model=None, batch_size=1)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                # csmt 接口
                trained_models=models_train(datasets_name,orig_models_name,X[label_ind.index, :], y[label_ind.index],X[label_ind.index, :], y[label_ind.index])
                y_test,y_pred=models_predict(trained_models,X[test_idx, :], y[test_idx])
                for i in range(0,len(orig_models_name)):
                    y_pred=np.argmax(y_pred[i], axis=1)
                accuracy = metrics.accuracy_score(y_test, y_pred)

                # Save intermediate results to file
                st = alibox.State(select_index=select_ind, performance=accuracy)
                saver.add_state(st)
                saver.save()

                # Passing the current progress to stopping criterion object
                stopping_criterion.update_information(saver)
            # Reset the progress in stopping criterion object
            stopping_criterion.reset()
            _result.append(copy.deepcopy(saver))
        return _result,label_ind
    
    # get the query results
    anal1 = ExperimentAnalyser(x_axis='num_of_queries')
    for _strategy in strategy_name_arr:
        result,label_ind=strategy_query(_strategy)
        anal1.add_method(_strategy,result)
    
    # anal1.plot_learning_curves(title='Learning curves', std_area=True,show=False)
    # plt.title('Learning curves',fontproperties='Times New Roman',fontsize=14)
    # plt.yticks(fontproperties='Times New Roman',fontsize=12)
    # plt.xticks(fontproperties='Times New Roman',fontsize=12)
    # plt.xlabel('Number of queries',fontproperties='Times New Roman',fontsize=14)
    # plt.ylabel('Performance',fontproperties='Times New Roman',fontsize=14)
    # plt.legend(loc=4,prop='Times New Roman')
    # plt.show()
    
    X_selection,y_selection=X[label_ind.index, :],y[label_ind.index]
    
    return X_selection,y_selection

    
def explain(model_,X,y):
    import time
    start=time.time()
    imputer = sage.MarginalImputer(model_,X)
    estimator = sage.PermutationEstimator(imputer, 'cross entropy')
    # estimator = sage.IteratedEstimator(imputer, 'cross entropy')
    sage_values = estimator(X,y)
    end=time.time()
    return sage_values.values,end-start

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm

    X,y,mask=get_raw_datasets(options)
    # mm=MinMaxScaler()
    # X=mm.fit_transform(X)
    # y=y.values
    X=X.values
    y=y.values.ravel()
    ## step1 feature selection
    X,y=feature_selection(X,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
    X_val,y_val=X_test,y_test
    
    ## step2 data selection
    X_selection,y_selection=data_selection(X,y)
    # X_selection,y_selection=X,y

    ## step3 model train
    trained_models=models_train(datasets_name,orig_models_name,X_selection,y_selection,X_selection,y_selection)
        y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)
    print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
    
    ## step4 model explain
    model_ = trained_models[0]
    X_explain,y_explain=X_selection,y_selection
    
    sage_values,time=explain(model_,X_explain,y_explain)
    print(sage_values)
    print(time)
    sage1=get_distribute(sage_values)
    
    sage_all_values,all_time=explain(model_,X[0:100],y[0:100])
    print(sage_all_values)
    print(all_time)
    sage2=get_distribute(sage_all_values)
    
    print(distEclud(sage1,sage2))


    
    
    

 



