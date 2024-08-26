import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
from csmt.data_validation.evidently import ColumnMapping
from csmt.data_validation.evidently.dashboard import Dashboard
from csmt.data_validation.evidently.dashboard.tabs import DataDriftTab
import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
import numpy as np

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm
    X,y,mask=get_raw_datasets(options)
    # X,y,_,_,_,_,mask=get_raw_datasets(options)
    # X=X.reshape(X.shape[0],-1)
    # X=pd.DataFrame(X)
    numerical_features=X.columns.values
    column_mapping = ColumnMapping()

    data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    
    X_0=X[y==0]
    X_1=X[y==1]
    # reference, current
    data_drift_dashboard.calculate(X_0, X_1, column_mapping=column_mapping)
    # data_drift_dashboard._save_to_json('zhs.json')
    
    data_drift_dashboard.save('data_analysis/experiment/flow_fsnet.html')