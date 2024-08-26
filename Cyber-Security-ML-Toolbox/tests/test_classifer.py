
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
import random
from csmt.figure import CFigure
from math import ceil

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)


# 目前仅限于异常检测模型
def plot_func(n_classes,models,models_name,table):
    n_classes=np.unique(y_train).size
    fig = CFigure(width=5 * len(models), height=5 * 2)
    for i in range(len(trained_models)):
        fig.subplot(2, int(ceil(len(models) / 2)), i + 1)
        fig.sp.plot_ds(X_test,y_test)
        fig.sp.plot_decision_regions(models[i], n_grid_points=100,n_classes=n_classes)
        fig.sp.plot_fun(models[i].predict_abnormal, plot_levels=False, 
                        multipoint=True, n_grid_points=50,alpha=0.6)
        fig.sp.title(models_name[i])
        fig.sp.text(0.01, 0.01, "Accuracy on test set: {:.2%}".format(table['accuracy'].tolist()[i]))
    fig.show()
    

def plot_headmap(X,a_score,model_name):
    X_x=X[:,0]
    X_y=X[:,1]
    plt.scatter(X_x, X_y, marker='o', c=a_score, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(model_name)
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
     
    #  plot_headmap(X_test,y_pred[0][:,0],orig_models_name)

     table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
    
     # 绘制决策边界-kitnet
    #  plot_func(np.unique(y_train).size,trained_models,orig_models_name,table)

     
    
 









