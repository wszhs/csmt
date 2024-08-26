import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import os
import sklearn
import sklearn.datasets
import sklearn.ensemble
import xgboost
import numpy as np
import pandas as pd

from csmt.Interpretability.omnixai.data.tabular import Tabular
from csmt.Interpretability.omnixai.preprocessing.tabular import TabularTransform
from csmt.Interpretability.omnixai.explainers.tabular import TabularExplainer
from csmt.Interpretability.omnixai.visualization.dashboard import Dashboard

# Load the dataset
feature_names = [
    "Age", "Workclass", "fnlwgt", "Education",
    "Education-Num", "Marital Status", "Occupation",
    "Relationship", "Race", "Sex", "Capital Gain",
    "Capital Loss", "Hours per week", "Country", "label"
]
feature_names_no_labels=["Age", "Workclass", "fnlwgt", "Education",
    "Education-Num", "Marital Status", "Occupation",
    "Relationship", "Race", "Sex", "Capital Gain",
    "Capital Loss", "Hours per week", "Country"]

df = pd.DataFrame(
    np.genfromtxt(os.path.join('/Users/zhanghangsheng/others_code/可解释包/OmniXAI/tutorials/data/', 'adult.data'), delimiter=', ', dtype=str),
    columns=feature_names
)
tabular_data = Tabular(
    data=df,
    categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
    target_column='label'
)
print(tabular_data)

# Train an XGBoost model
np.random.seed(1)
transformer = TabularTransform().fit(tabular_data)
class_names = transformer.class_names
x = transformer.transform(tabular_data)
train, test, labels_train, labels_test = \
    sklearn.model_selection.train_test_split(x[:, :-1], x[:, -1], train_size=0.80)
print('Training data shape: {}'.format(train.shape))
print('Test data shape:     {}'.format(test.shape))

gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(train, labels_train)
print('Test accuracy: {}'.format(
    sklearn.metrics.accuracy_score(labels_test, gbtree.predict(test))))

preprocess = lambda z: transformer.transform(z)

# Initialize a TabularExplainer
explainers = TabularExplainer(
    explainers=["lime", "shap", "pdp"],
    mode="classification",
    data=transformer.invert(train),
    model=gbtree,
    preprocess=preprocess,
    params={
        "lime": {"kernel_width": 3},
        "shap": {"nsamples": 100}
    }
)
# Apply an inverse transform, i.e., converting the numpy array back to `Tabular`
test_instances = transformer.invert(test[1653:1658])
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