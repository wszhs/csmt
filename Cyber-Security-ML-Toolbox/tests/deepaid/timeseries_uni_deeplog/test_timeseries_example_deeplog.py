import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

# Train Deeplog anomaly detection model
import numpy as np
import torch
from deeplog import train_deeplog, test_deeplog
train_data = np.load('tests/deepaid/timeseries_uni_deeplog/data/train_data.npz')
train_normal_seq = train_data['train_normal_seq']
train_normal_label = train_data['train_normal_label']
# model = train_deeplog(train_normal_seq, train_normal_label)
# torch.save(model, 'tests/deepaid/timeseries_uni_deeplog/save/LSTM_onehot.pth.tar')

model = torch.load('tests/deepaid/timeseries_uni_deeplog/save/LSTM_onehot.pth.tar')
# Validate the performance of trained model
test_normal_loader = np.load('tests/deepaid/timeseries_uni_deeplog/data/test_normal_loader.npy',allow_pickle=True)
test_abnormal_loader = np.load('tests/deepaid/timeseries_uni_deeplog/data/test_abnormal_loader.npy',allow_pickle=True)
test_deeplog(model, test_normal_loader, test_abnormal_loader)

"""Step 1: Load your model"""
from deeplog import LSTM_onehot
import torch

"""Step 2: Find an anomaly you are interested in"""
from csmt.Interpretability.deepaid.utils import deeplogtools_seqformat
abnormal_data = np.load('tests/deepaid/timeseries_uni_deeplog/data/abnormal_data.npy')
idx = 100
seq, label, anomaly_timeseries = deeplogtools_seqformat(model, abnormal_data, num_candidates=9, index=idx)
# print(seq.shape,label.shape)

"""Step 3: Create a DeepAID Interpreter"""
from csmt.Interpretability.deepaid.interpreters.timeseries_onehot import UniTimeseriesAID
feature_desc = np.load('tests/deepaid/timeseries_uni_deeplog/data/log_key_meanning.npy') # feature_description
my_interpreter = UniTimeseriesAID(model, feature_desc=feature_desc, class_num=28)

"""Step 4: Interpret your anomaly and show the result"""
interpretation = my_interpreter(seq, label)
my_interpreter.show_table(anomaly_timeseries, interpretation)