import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from pandas.core import base
import matplotlib
import csmt.Interpretability.sage as sage
import os
import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
import seaborn as sns
from csmt.Interpretability.rexplain import removal, behavior, summary

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
    
    # Make model callable with numpy inputs
    model_lam = lambda x: trained_models[0].predict(x)

    # # Model extension
    # extension=removal.DefaultExtension(X_test.values[0], model_lam)
    extension = removal.MarginalExtension(X_test[:100], model_lam)
    # Cooperative game
    game = behavior.PredictionGame(extension,X_test[0])
    # Summary technique
    attr = summary.RemoveIndividual(game)
#     attr=summary.ShapleyValue(game)
    print(attr[:,0])
    
    plt.figure(figsize=(9, 6))
    plt.bar(np.arange(len(attr)), attr[:,0])
    plt.xticks(np.arange(len(attr)),
            rotation=45, rotation_mode='anchor',
            ha='right', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.title('Remove individual', fontsize=20)
    plt.tight_layout()
    plt.show()