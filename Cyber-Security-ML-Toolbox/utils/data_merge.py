'''
Author: your name
Date: 2021-04-23 12:09:36
LastEditTime: 2021-05-09 14:34:16
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_data_pre.py
'''
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)

import pandas as pd
spl_file_path='csmt/datasets/data/datacon-log/datacon_spl.csv'
tls_file_path='csmt/datasets/data/datacon-log/datacon_tls.csv'
flowmeter_file_path='csmt/datasets/data/datacon-log/datacon_flowmeter.csv'
file_out='csmt/datasets/data/datacon-log/datacon_all.csv'

df_spl = pd.read_csv(spl_file_path, encoding='utf8', low_memory=False)
df_tls=pd.read_csv(tls_file_path, encoding='utf8', low_memory=False)
df_flowmeter=pd.read_csv(flowmeter_file_path, encoding='utf8', low_memory=False)
print(df_spl.shape)
print(df_tls.shape)
print(df_flowmeter.shape)

df = pd.merge(df_spl, df_tls,how = "inner",on = "uid")
df= pd.merge(df, df_flowmeter,how = "inner",on = "uid")

df.to_csv(file_out, index=False, encoding="utf-8")
