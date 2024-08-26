# Train an autoencoder-based DL model
import numpy as np
import torch
from autoencoder import train, test, test_plot
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.Interpretability.deepaid.utils import validate_by_rmse, Normalizer
train_feat = np.load('tests/deepaid/tabular_kitsune/data/train_benign_feat.npy')
normer = Normalizer(train_feat.shape[-1],online_minmax=True)
train_feat = normer.fit_transform(train_feat)
# model, thres = train(train_feat, train_feat.shape[-1])
# torch.save({'net':model,'thres':thres},'tests/deepaid/tabular_synthesis/save/autoencoder.pth.tar')

model_dict = torch.load('tests/deepaid/tabular_synthesis/save/autoencoder.pth.tar')
model = model_dict['net']
thres = model_dict['thres']

test_feat = np.load('tests/deepaid/tabular_kitsune/data/test_mirai_ddos.npy')
test_feat = normer.transform(test_feat)
# print(test_feat.shape)
rmse_vec = test(model,thres,test_feat)
# print(rmse_vec.shape)
# print(thres.shape)
test_plot(test_feat, rmse_vec, thres) # ACC = 0.84

"""Step 1: Load your model"""
# from autoencoder import autoencoder
# from csmt.Interpretability.deepaid.utils import Normalizer

# """Step 2: Find an anomaly you are interested in (here we use the IP scan in Mirai botnet attack)"""
# anomaly = test_feat[np.argsort(rmse_vec)[-100]]

"""Step 3: Create a DeepAID Tabular Interpreter"""
# from csmt.Interpretability.deepaid.interpreters.tabular import TabularAID
# feature_desc = np.load('tests/deepaid/tabular_kitsune/data/AI_feature_name.npy') # feature_description
# my_interpreter = TabularAID(model,thres,input_size=100,feature_desc=feature_desc)

"""Step 4: Interpret your anomaly and show the result"""
# interpretation = my_interpreter(anomaly)
# # DeepAID supports three kinds of visualization of results:
# my_interpreter.show_table(anomaly,interpretation, normer) 
# my_interpreter.show_plot(anomaly, interpretation, normer)
# my_interpreter.show_heatmap(anomaly,interpretation, normer)