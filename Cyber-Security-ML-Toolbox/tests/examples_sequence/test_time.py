import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.datasets.sequence.time_series_data import *
# print(get_UCR_univariate_list())
# print(get_UCR_multivariate_list())

X,y,mask=load_natops()
print(X.shape)