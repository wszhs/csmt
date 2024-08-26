import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random 
import numpy as np

# Generate Gaussian blobs (single cluster) consists 5050 100-dimension samples
X, _ = make_blobs(n_samples=5050, centers=1, n_features=100,
                  random_state=0, cluster_std=2.)
print('X.shape:',X.shape)

# first 5000 samples are used for training
X_train = X[:-50,:]

# generate anomalies by randomly perturbing 10 dimensions (1%) in the last 50 samples
random.seed(0)
noise_idx = random.choices(list(range(100)),k=10)
noise_idx.sort()
print('perturb index:',noise_idx)
noise_data = np.random.uniform(-20,-20,(50,10))
X_anomaly = X[-50:,:]
X_anomaly[:,noise_idx] += noise_data

# visualize anomaly and normal training data in 2D space 
# X_train_plot = X_train[:, noise_idx]
# pca = PCA(n_components=2).fit(X_train_plot)
# plt.scatter(X_train_plot[:, 0], X_train_plot[:, 1], alpha=0.5, s=3, label="training data (normal)")
# X_anomaly_plot = X_anomaly[:,noise_idx]
# pca = PCA(n_components=2).fit(X_anomaly_plot)
# plt.scatter(X_anomaly_plot[:, 0], X_anomaly_plot[:, 1], alpha=1., s=4, c="r", label="anomaly")
# plt.legend()
# plt.show()

# Train an autoencoder-based DL model
import numpy as np
import torch
import sys
from autoencoder import train, test, test_plot
from csmt.Interpretability.deepaid.utils import validate_by_rmse, Normalizer
normer = Normalizer(X_train.shape[-1],online_minmax=False)
X_train = normer.fit_transform(X_train)
model, thres = train(X_train, X_train.shape[-1])
torch.save({'net':model,'thres':thres},'tests/deepaid/tabular_synthesis/save/autoencoder.pth.tar')

# Validate the performance of trained model
X_anomaly_norm = normer.transform(X_anomaly)
rmse_vec = test(model,thres,X_anomaly_norm)
test_plot(X_anomaly_norm, rmse_vec, thres) # ACC = 100%

# """Load the model"""
# from autoencoder import autoencoder
# from csmt.Interpretability.deepaid.utils import Normalizer
# model_dict = torch.load('tests/deepaid/tabular_synthesis/save/autoencoder.pth.tar')
# model = model_dict['net']
# thres = model_dict['thres']

# """ Create a DeepAID Tabular Interpreter"""
# from csmt.Interpretability.deepaid.interpreters.tabular import TabularAID
# my_interpreter = TabularAID(model,thres,input_size=100,k=10,steps=100,auto_params=False)

# """Interpret the anomalies"""
# # for anomaly in X_anomaly:
# anomaly = X_anomaly[5]
# interpretation = my_interpreter(anomaly)
# my_interpreter.show_table(anomaly,interpretation, normer) 

# print('perturb index:',noise_idx)
