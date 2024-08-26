import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import copy
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from csmt.active_learning.alipy import ToolBox
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets

from csmt.active_learning.modAL.density import information_density
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 

arguments = sys.argv[1:]
options = parse_arguments(arguments)
X,y,mask=get_raw_datasets(options)

X= TSNE(n_components=2).fit_transform(X)
# X= PCA(n_components=2).fit_transform(X)

cosine_density = information_density(X, 'cosine')

euclidean_density = information_density(X, 'euclidean')

# visualizing the cosine and euclidean information density
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=cosine_density, cmap='viridis', s=50)
    plt.title('The cosine information density')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=euclidean_density, cmap='viridis', s=50)
    plt.title('The euclidean information density')
    plt.colorbar()
    plt.show()
