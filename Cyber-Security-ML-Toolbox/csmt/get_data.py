
from csmt.datasets import *
from csmt.normalizer import Normalizer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from csmt.decomposition.decomposition import tsne_dim_redu,pca_dim_redu,svd_dim_redu
from sklearn.model_selection import train_test_split
from csmt.classifiers.graph.dataset import Dataset,CogDLDataset
import numpy as np
from csmt.config import CSMT_NUMPY_DTYPE

def get_graph_cogdl_datasets(options):
    datasets_name=options.datasets
    data=globals().get('load_'+datasets_name)()
    return data

def get_graph_datasets(options): 
    datasets_name=options.datasets
    features, adj, labels, split_ids=globals().get('load_'+datasets_name)()
    return features, adj, labels, split_ids

def get_graph_grb_datasets(options):
    dataset_name=options.datasets
    if 'grb' in dataset_name:
        data = Dataset(name=dataset_name, 
                    data_dir="./data/",
                    mode="full",
                    feat_norm="arctan")
    else:
        data=CogDLDataset(name=dataset_name)
    return data



def get_raw_datasets(options):
    datasets_name=options.datasets
    train_test_dataset={'nslkdd','mnist','flow_fsnet','flow_mampf','wfa','pcap','imdb'}
    if datasets_name in train_test_dataset:
        X_train,y_train,X_val,y_val,X_test,y_test,constraints=globals().get('load_'+datasets_name)()
        return X_train,y_train,X_val,y_val,X_test,y_test,constraints
    X,y,mask=globals().get('load_'+datasets_name)()
    X=X.astype(CSMT_NUMPY_DTYPE)
    return X,y,mask


def get_datasets(datasets_name):
    train_test_dataset={'nslkdd','mnist','flow_fsnet','flow_mampf','wfa','pcap','imdb'}
    if datasets_name in train_test_dataset:
        X_train,y_train,X_val,y_val,X_test,y_test,constraints=globals().get('load_'+datasets_name)()
        return X_train,y_train,X_val,y_val,X_test,y_test,constraints

    X,y,constraints=globals().get('load_'+datasets_name)()
    normer = Normalizer(X.shape[-1],online_minmax=False)
    X = normer.fit_transform(X)

    return pre_processing(X,y,constraints)

def pre_processing(X,y,constraints):
    # mm=MinMaxScaler()
    # X=mm.fit_transform(X)
    # X_train,y_train,X_val,y_val,X_test,y_test=train_val_test_split(X,y,0.6,0.3,0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
    X_val,y_val=X_test,y_test
    
    X_train=X_train.astype(CSMT_NUMPY_DTYPE)
    X_test=X_test.astype(CSMT_NUMPY_DTYPE)
    X_val=X_val.astype(CSMT_NUMPY_DTYPE)
    # y_train=y_train.astype(CSMT_NUMPY_DTYPE)
    # y_test=y_test.astype(CSMT_NUMPY_DTYPE)
    # y_val=y_val.astype(CSMT_NUMPY_DTYPE)

    if type(y_train) is not np.ndarray:
        X_train,y_train,X_val,y_val,X_test,y_test=X_train.values,y_train.values,X_val.values,y_val.values,X_test.values,y_test.values

    return X_train,y_train,X_val,y_val,X_test,y_test,constraints
