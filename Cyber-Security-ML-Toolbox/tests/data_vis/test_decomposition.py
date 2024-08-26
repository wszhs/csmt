'''
Author: your name
Date: 2021-05-14 17:05:47
LastEditTime: 2021-07-20 14:54:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/data_vis/test_TSNE.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from csmt.get_model_data import get_raw_datasets,parse_arguments
from csmt.figure.visualml.plot_ds import plot_ds_2d,plot_ds_3d
 
arguments = sys.argv[1:]
options = parse_arguments(arguments)
datasets_name=options.datasets
orig_models_name=options.algorithms
X,y,mask=get_raw_datasets(options)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 

# mm=MinMaxScaler()
# X=mm.fit_transform(X)
 
# X_de = TSNE(n_components=2).fit_transform(X) 
X_de= PCA(n_components=2).fit_transform(X)

df = pd.DataFrame(data = X_de, columns = ['comp0', 'comp1'])
df['label']=y.values
sns.jointplot(x='comp0', y='comp1', data=df,hue='label')

plt.show()

# plot_ds_2d(X_de,y)


# X_de_data = np.vstack((X_de.T, y)).T 
# df_de = pd.DataFrame(X_de_data, columns=['Dim1', 'Dim2', 'class']) 
# df_de.head()
# plt.figure(figsize=(8, 8)) 
# sns.scatterplot(data=df_de, hue='class', x='Dim1', y='Dim2') 
# plt.show()
