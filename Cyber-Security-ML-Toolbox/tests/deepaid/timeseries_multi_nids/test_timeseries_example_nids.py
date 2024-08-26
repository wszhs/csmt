# Train Deeplog anomaly detection model
import numpy as np
import torch
from lstm_multivariate import train, test, test_from_iter
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
train_feat = np.load('tests/deepaid/timeseries_multi_nids/data/train_feat.npy')
from csmt.Interpretability.deepaid.utils import validate_by_rmse, Normalizer
normer = Normalizer(train_feat.shape[-1],online_minmax=False)
train_feat = normer.fit_transform(train_feat)
print(train_feat.shape)
# model, thres = train(train_feat)
# torch.save({'net':model,'thres':thres},'tests/deepaid/timeseries_multi_nids/save/lstm_multivariate.pth.tar')

model_dict = torch.load('tests/deepaid/timeseries_multi_nids/save/lstm_multivariate.pth.tar')
model = model_dict['net']
thres = model_dict['thres']

import matplotlib.pyplot as plt
test_feat = np.load('tests/deepaid/timeseries_multi_nids/data/test_feat_cicddos.npy')
test_label = np.load('tests/deepaid/timeseries_multi_nids/data/test_label_cicddos.npy')
# import pandas as pd
# print(pd.DataFrame(test_label).value_counts())

test_feat = normer.transform(test_feat)
rmse_vec = test(model,thres, test_feat)
print(rmse_vec)
plt.scatter(np.linspace(0,len(test_feat)-1,len(test_feat)),rmse_vec,s=3)  
plt.plot(np.linspace(0,len(test_feat)-1,len(test_feat)),[thres]*len(test_feat),c='black')
plt.show()
pred = validate_by_rmse(rmse_vec,thres,test_label)
# print(pred)


# """Step 1: Load your model"""
# from lstm_multivariate import LSTM_multivariate
# from csmt.Interpretability.deepaid.utils import Normalizer

# """Step 2: Find an anomaly you are interested in"""
# anomaly = test_feat[np.argsort(rmse_vec)[-100]]
# from csmt.Interpretability.deepaid.utils import multiLSTM_seqformat
# idx = 100
# seq_feat, interp_feat = multiLSTM_seqformat(test_feat, seq_len = 5, index=idx)

# """Step 3: Create a DeepAID multivariate Time-Series Interpreter"""

# from csmt.Interpretability.deepaid.interpreters.timeseries_multivariate import MultiTimeseriesAID 
# feature_desc = np.load('tests/deepaid/timeseries_multi_nids/data/AI_feature_name.npy') # feature_description

# my_interpreter = MultiTimeseriesAID(model,thres,input_size=100,feature_desc=feature_desc)

# """Step 4: Interpret your anomaly and show the result"""
# interpretation = my_interpreter(seq_feat)
# my_interpreter.show_table(interp_feat,interpretation, normer)
# my_interpreter.show_plot(interp_feat, interpretation, normer)
# my_interpreter.show_heatmap(interp_feat,interpretation, normer)
# print(interpretation)