import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import csmt.Interpretability.sage as sage

import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
import torch
import seaborn as sns
import csmt.Interpretability.shap as shap
from csmt.feature_selection import selectGa
# from xgboost import plot_importance
from csmt.figure.visualml.plot_importance import plot_xg_importance,plot_feature_importance_all,plot_dot


def plot_heatmap(table):
    matplotlib.style.use('seaborn-whitegrid')
    # table = np.random.rand(10, 12)
    sns.heatmap(table,vmin=0,vmax=1, cmap='viridis',annot=True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14,rotation=0, horizontalalignment= 'right')
    plt.show()

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name='breast_cancer_zhs'
    orig_models_name=['mlp_torch']
    evasion_algorithm=options.evasion_algorithm

    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(datasets_name)

    trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)

    y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)
    print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
    
    
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    # explain predictions of the model on four images
    explainer = shap.DeepExplainer(trained_models[0].classifier.model, X_train)
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = explainer.shap_values(X_test)
    
    


