import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
from csmt.figure.visualml.plot_ds import plot_ds_2d,plot_ds_3d
 

arguments = sys.argv[1:]
options = parse_arguments(arguments)
datasets_name=options.datasets
orig_models_name=options.algorithms

# X,y,_,_,_,_,mask=get_raw_datasets(options)
# X=X.reshape(X.shape[0],-1)

X,y,mask=get_raw_datasets(options)
X_1=X[y==1]
X_0=X[y==0]

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
 
# X_de = TSNE(n_components=1).fit_transform(X) 
X_de_0 = PCA(n_components=1).fit_transform(X_0)/100000000.0
X_de_1 = PCA(n_components=1).fit_transform(X_1)/100000000.0
# X_de=X.iloc[:,10]
# mm=MinMaxScaler()
# X_de_0=mm.fit_transform(X_de_0)
# X_de_1=mm.fit_transform(X_de_1)

def plot_vec(X_0,X_1):
    fig=plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1)
    n, bins, patches=plt.hist(X_0, bins=200, color='blue',label='Normal')
    # for i in range(len(n)):
    #     if n[i]>500:
    #         plt.text(bins[i], n[i], int(n[i]), fontsize=8, horizontalalignment="center")
    n, bins, patches=plt.hist(X_1, bins=200, color='red',label='Abnormal')
    for i in range(len(n)):
        if n[i]>5:
            plt.text(bins[i], n[i], int(n[i]), fontsize=5, horizontalalignment="center")
    # plt.legend('zzzz')
    my_x_ticks = np.arange(-1, 5, 0.5)
    plt.xticks(my_x_ticks)
    plt.yticks(fontsize=30,rotation=0, horizontalalignment= 'right',)
    plt.xticks(fontsize=25,rotation=45, horizontalalignment= 'right',)
    plt.xlabel('', fontsize=25)
    plt.ylabel('Number of Data Points', fontsize=25)
    # ax.set_yscale("log")
    plt.yscale('symlog')
    # plt.xscale('symlog')
    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.65,1),prop={'size':20},edgecolor='black')
    plt.show()
# plot_vec(X_de_0,X_de_1)

def plot_vec2(X_0,X_1):
    plt.figure(figsize=(10, 10))
    plt.scatter(np.linspace(0,len(X_0)-1,len(X_0)),X_0,s=1,c='blue',label='normal')
    plt.scatter(np.linspace(0,len(X_1)-1,len(X_1)),X_1,s=1,c='red',label='Abnormal')
    plt.yticks(fontsize=25,rotation=0, horizontalalignment= 'right',)
    plt.xticks(fontsize=20,rotation=30, horizontalalignment= 'right',)
    my_y_ticks = np.arange(-1, 5, 0.5)
    plt.yticks(my_y_ticks)
    plt.gca().set_xlabel('Data Point',fontdict={'size':25})
    plt.gca().set_ylabel('Feature values after reducing to one dimension',fontdict={'size':20})
    # plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.7,1),prop={'size':15},edgecolor='black')
    plt.show()

plot_vec2(X_de_0,X_de_1)