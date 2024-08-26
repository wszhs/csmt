import sys

from sqlalchemy import Constraint
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

import os
import numpy as np
from csmt.Interpretability.omnixai.data.tabular import Tabular
from csmt.Interpretability.omnixai.explainers.data import DataAnalyzer
from csmt.Interpretability.omnixai.visualization.dashboard import Dashboard


from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
import pandas as pd
arguments = sys.argv[1:]
options = parse_arguments(arguments)
X,y,constraints=get_raw_datasets(options)
df= pd.concat([X,y],axis=1)

feature_names=[]
for i in range(X.shape[1]):
    feature_names.append('F'+str(i))
feature_names.append('label')
    
df.columns=feature_names

tabular_data = Tabular(
    df.values,
    feature_columns=feature_names,
    target_column='label'
)

print(type(tabular_data))
print(tabular_data)

explainer = DataAnalyzer(
    explainers=["correlation", "imbalance#0", "imbalance#1", "imbalance#2", "mutual", "chi2"],
    data=tabular_data
)
# Generate explanations by calling `explain_global`.
explanations = explainer.explain_global(
    params={"imbalance#0": {"features": ["F0"]},
            "imbalance#1": {"features": ["F1"]},
            "imbalance#2": {"features": ["F0", "F1"]}}
)

# Launch a dashboard for visualization.
dashboard = Dashboard(global_explanations=explanations)
dashboard.show()
