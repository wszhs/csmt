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

df=X.iloc[:,0:5]
df_new=df.copy()

df_new['label']=y

sns.set_palette(['blue', 'red'])
g = sns.PairGrid(df_new, hue="label")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


plt.show()



