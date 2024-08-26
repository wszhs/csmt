import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
import numpy as np
from csmt.get_model_data import get_datasets,models_train,parse_arguments,models_train,print_results,models_predict
from csmt.attacks.evasion.evasion_attack import EvasionAttack
import torch
import matplotlib
import matplotlib.pyplot as plt
import random
from csmt.figure import CFigure
from math import ceil
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def plot_fun_path(n_classes,trained_models,orig_models_name,table,adv_path):
    fig = CFigure(width=5 * len(trained_models), height=5 * 2)

    for i in range(len(trained_models)):
        fig.subplot(2, int(ceil(len(trained_models) / 2)), i + 1)
        fig.sp.plot_ds(X_test,y_test)
        fig.sp.plot_decision_regions(trained_models[i], n_grid_points=100,n_classes=n_classes)
        for j in range(0,10):
            adv_path=X_adv_path[j,:]
            fig.sp.plot_path(adv_path)
    fig.show()


if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    attack_model_name=options.attack_models
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm
    X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)

    setup_seed(20)
    attack_models=models_train(datasets_name,attack_model_name,X_train,y_train,X_val,y_val)
    setup_seed(20)
    trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)
    y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)
    
    X_test_1=X_test[y_test==1]
    y_test_1=y_test[y_test==1]
    X_test_0=X_test[y_test==0]
    y_test_0=y_test[y_test==0]

    table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
    

    X_adv,y_adv,X_adv_path=EvasionAttack(attack_models,evasion_algorithm,X_test_1,y_test_1,constraints)

    X_test_adv = np.append(X_test_0, X_adv, axis=0)
    y_test_adv = np.append(y_test_0, y_adv, axis=0)

    y_test_adv,y_pred_adv=models_predict(trained_models,X_test_adv,y_test_adv)

    print_results(datasets_name,orig_models_name,y_test_adv,y_pred_adv,'adversarial_accuracy')

    plot_fun_path(np.unique(y_train).size,trained_models,orig_models_name,table,X_adv_path)


