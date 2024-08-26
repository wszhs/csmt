import sys
from sqlalchemy import Constraint
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

import os
import numpy as np
from csmt.Interpretability.omnixai.data.tabular import Tabular
from csmt.Interpretability.omnixai.explainers.data import DataAnalyzer
from csmt.Interpretability.omnixai.visualization.dashboard import Dashboard
from csmt.Interpretability.omnixai.preprocessing.tabular import TabularTransform
from csmt.Interpretability.omnixai.explainers.tabular import TabularExplainer

import sklearn
import sklearn.datasets
import sklearn.ensemble
from sklearn.linear_model import LogisticRegression
import xgboost

from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
import pandas as pd
arguments = sys.argv[1:]
options = parse_arguments(arguments)
X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)
X=np.r_[X_train,X_val,X_test]
y=np.r_[y_train,y_val,y_test]

X=pd.DataFrame(X)
y=pd.DataFrame(y)
df= pd.concat([X,y],axis=1)

feature_names=[]
feature_names_no_labels=[]
for i in range(X.shape[1]):
    feature_names.append('F'+str(i))
    feature_names_no_labels.append('F'+str(i))
feature_names.append('label')
    
df.columns=feature_names

tabular_data = Tabular(
    df.values,
    feature_columns=feature_names,
    target_column='label'
)

transformer = TabularTransform().fit(tabular_data)
class_names = transformer.class_names

model=LogisticRegression()
model.fit(X_train, y_train)


print('Test accuracy: {}'.format(
    sklearn.metrics.accuracy_score(y_test, model.predict(X_test))))
# 将本格式转为numpy格式
preprocess = lambda z: transformer.transform(z)

# Initialize a TabularExplainer
explainers = TabularExplainer(
    explainers=["lime", "shap", "pdp"],
    mode="classification",
    data=transformer.invert(X_train),
    model=model,
    preprocess=preprocess,
    params={
        "lime": {"kernel_width": 3},
        "shap": {"nsamples": 100}
    }
)
# Apply an inverse transform, i.e., converting the numpy array back to `Tabular`
test_instances = transformer.invert(X_test[0:10])
# Generate explanations
local_explanations = explainers.explain(X=test_instances)
global_explanations = explainers.explain_global()

# Launch a dashboard for visualization
dashboard = Dashboard(
    instances=test_instances,
    local_explanations=local_explanations,
    global_explanations=global_explanations,
    class_names=class_names,
    params={"pdp": {"features": feature_names_no_labels}}
)
dashboard.show()