
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import plotly.io as pio
pio.renderers.default = "png"

import os
import numpy as np
import pandas as pd
from csmt.Interpretability.omnixai.data.timeseries import Timeseries
from csmt.Interpretability.omnixai.explainers.timeseries import TimeseriesExplainer

# Load the time series dataset
df = pd.read_csv(os.path.join("/Users/zhanghangsheng/others_code/可解释包/OmniXAI/tutorials/data/", "timeseries.csv"))
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
df = df.rename(columns={"horizontal": "values"})
df = df.set_index("timestamp")
df = df.drop(columns=["anomaly"])
print(df)

# Split the dataset into training and test splits
train_df = df.iloc[:9150]
test_df = df.iloc[9150:9300]
# A simple threshold for detecting anomaly data points
threshold = np.percentile(train_df["values"].values, 90)

# A simple detector for determining whether a window of time series is anomalous
def detector(ts: Timeseries):
    scores = []
    for x in ts.values:
        anomaly_scores = np.sum((x > threshold).astype(int))
        scores.append(anomaly_scores / x.shape[0])
    return np.array(scores)

# Initialize a TimeseriesExplainer
explainers = TimeseriesExplainer(
    explainers=["shap"],
    mode="anomaly_detection",
    data=Timeseries.from_pd(train_df),
    model=detector,
    preprocess=None,
    postprocess=None
)
# Generate explanations
test_instances = Timeseries.from_pd(test_df)
local_explanations = explainers.explain(test_instances)

# result=detector(test_df)

from csmt.Interpretability.omnixai.visualization.dashboard import Dashboard
dashboard = Dashboard(instances=test_instances, local_explanations=local_explanations)
dashboard.show()