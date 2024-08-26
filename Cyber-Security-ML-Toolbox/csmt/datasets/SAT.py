from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from typing import BinaryIO
import pandas as pd
import numpy as np
from os import path
ROOT_PATH='D:/STIN/MIX_NET'

def datasetprocessing(data_class_type):
    if data_class_type=='single_binary' or data_class_type=='multi':
        raise Exception('Data classification type is not supported！')
    file_path=path.join(ROOT_PATH,'MIX_NET.csv')

    label_map = {'BENIGN': 0, 'Syn_DDoS': 1, 'UDP_DDoS': 1, 'Botnet': 1,'Web Attack': 1, 'Backdoor': 1}

    # 读取csv文件
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)

    # 删除第一列序号，无用信息
    df = df.drop([df.columns[0]], axis=1)

    # 删除字符串左侧空格
    df.columns = df.columns.str.lstrip()
   
    # 删除空值
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    # 删除LDAP_DDoS、MSSQL_DDoS、NetBIOS_DDoS、Portmap_DDoS
    df = df[~df['Label'].isin(['LDAP_DDoS','MSSQL_DDoS','NetBIOS_DDoS','Portmap_DDoS'])]
    
    # 提取BENIGN
    benign = df[df['Label'] == 'BENIGN']
    benign1 = benign.sample(n = 55000,random_state = 1,axis = 0)  ##采样55000条BENIGN
    benign2 = benign.sample(n = 13000,random_state = 1,axis = 0)  ##采样13000条BENIGN
    benign3 = benign.sample(n = 15000,random_state = 1,axis = 0)  ##采样15000条BENIGN
    
    # ##1.SYN_DDoS+BENIGN
    syn_ddos = df[df['Label'] == 'Syn_DDoS']
    #print(syn_ddos.shape)   ##(54789, 31)
    syn_benign = pd.concat([syn_ddos,benign1]) 
    syn_benign['Label'] = syn_benign['Label'].map(label_map)
    X = syn_benign.drop(['Label'], axis=1)
    y = syn_benign['Label']

 
    
    # ##2.UDP_DDoS+BENIGN
    # udp_ddos = df[df['Label'] == 'UDP_DDoS']
    # #print(udp_ddos.shape)        ##(57082, 31)
    # udp_benign = pd.concat([udp_ddos,benign1])
    # udp_benign['Label'] = udp_benign['Label'].map(label_map)
    # X = udp_benign.drop(['Label'], axis=1)
    # y = udp_benign['Label']

    
    # ##3.Botnet+BENIGN
    # botnet = df[df['Label'] == 'Botnet']
    # #print(botnet.shape)        ##(14622, 31)
    # botnet_benign = pd.concat([botnet,benign3])
    # botnet_benign['Label'] = botnet_benign['Label'].map(label_map)
    # X = botnet_benign.drop(['Label'], axis=1)
    # y = botnet_benign['Label']
    

    # ##4.Web Attack+BENIGN
    # web_attack = df[df['Label'] == 'Web Attack']
    # #print(web_attack.shape)        ##(13017, 31)
    # web_benign = pd.concat([web_attack,benign2])
    # web_benign['Label'] = web_benign['Label'].map(label_map)
    # X = web_benign.drop(['Label'], axis=1)
    # y = web_benign['Label']
    

    # ##5.Backdoor+BENIGN
    # backdoor = df[df['Label'] == 'Backdoor']
    # # print(backdoor.shape)        ##(12762, 31)
    # backdoor_benign = pd.concat([backdoor,benign2])
    # backdoor_benign['Label'] = backdoor_benign['Label'].map(label_map)
    # X = backdoor_benign.drop(['Label'], axis=1)
    # y = backdoor_benign['Label']
    

    print(X.shape)
    print(y.shape)
    return X,y
datasetprocessing('csv')
