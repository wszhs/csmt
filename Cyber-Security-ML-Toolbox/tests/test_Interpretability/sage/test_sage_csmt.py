import sys

from pandas.core import base
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import csmt.Interpretability.sage as sage
import os

import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
import random
import torch
import seaborn as sns
import csmt.Interpretability.shap as shap
from csmt.figure.visualml.plot_importance import plot_vec

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)

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
    model_lam = trained_models[0]

    X_test=X_test[0:500]
    y_test=y_test[0:500]
    
    imputer = sage.MarginalImputer(model_lam,X_test)
    # estimator = sage.PermutationEstimator(imputer, 'cross entropy')
    estimator = sage.KernelEstimator(imputer, 'cross entropy')
    # estimator = sage.IteratedEstimator(imputer, 'cross entropy')
    sage_values = estimator(X_test, y_test)

    attr=sage_values.values
    
    print(attr)
    
    plot_vec(attr,'SAGE')




