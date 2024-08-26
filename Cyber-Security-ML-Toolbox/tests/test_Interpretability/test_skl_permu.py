import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import csmt.Interpretability.sage as sage

import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
from csmt.figure.visualml.plot_importance import plot_xg_importance,plot_feature_importance_all,plot_dot,plot_vec
from sklearn.inspection import permutation_importance

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm

    X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)

    trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)

    y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)
    print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
    
    result = permutation_importance(trained_models[0].classifier.model, X_train, y_train, n_repeats=10,random_state=0)
    
#     print(result.importances_mean)
    # print(result.importances_std)
    
    plot_vec(result.importances_mean,'permutation_importance')
