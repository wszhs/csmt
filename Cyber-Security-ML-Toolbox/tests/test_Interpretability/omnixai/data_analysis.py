import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import plotly.io as pio
pio.renderers.default = "png"

import os
import numpy as np
from csmt.Interpretability.omnixai.data.tabular import Tabular
from csmt.Interpretability.omnixai.explainers.data import DataAnalyzer
from csmt.Interpretability.omnixai.visualization.dashboard import Dashboard

# Load the dataset to analyze
data = np.genfromtxt(os.path.join('/Users/zhanghangsheng/others_code/可解释包/OmniXAI/tutorials/data/', 'adult.data'), delimiter=', ', dtype=str)
# The column names for this dataset
feature_names = [
    "Age", "Workclass", "fnlwgt", "Education",
    "Education-Num", "Marital Status", "Occupation",
    "Relationship", "Race", "Sex", "Capital Gain",
    "Capital Loss", "Hours per week", "Country", "label"
]
# Construct a `Tabular` instance for this tabular dataset, 
# e.g., specifying feature columns, categorical feature names and target column name.
tabular_data = Tabular(
    data,
    feature_columns=feature_names,
    categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
    target_column='label'
)

explainer = DataAnalyzer(
    explainers=["correlation", "imbalance#0", "imbalance#1", "imbalance#2", "imbalance#3", "mutual", "chi2"],
    data=tabular_data
)
# Generate explanations by calling `explain_global`.
explanations = explainer.explain_global(
    params={"imbalance#0": {"features": ["Sex"]},
            "imbalance#1": {"features": ["Race"]},
            "imbalance#2": {"features": ["Sex", "Race"]},
            "imbalance#3": {"features": ["Marital Status", "Age"]}}
)

# Launch a dashboard for visualization.
dashboard = Dashboard(global_explanations=explanations)
dashboard.show()