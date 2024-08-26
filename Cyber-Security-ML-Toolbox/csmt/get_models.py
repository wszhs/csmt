from csmt.classifiers.classic.omni import Omni
from csmt.classifiers.classic.adaboost import Adaboost
from csmt.classifiers.classic.decision_tree import DecisionTree
from csmt.classifiers.classic.k_nearest_neighbours import KNearestNeighbours
from csmt.classifiers.classic.logistic_regression import LogisticRegression
from csmt.classifiers.classic.random_forest import RandomForest
from csmt.classifiers.classic.support_vector_machine import SupportVectorMachine
from csmt.classifiers.classic.naive_bayes import NaiveBayes
from csmt.classifiers.classic.xgboost import XGBoost
from csmt.classifiers.classic.lightgbm import LightGBM
from csmt.classifiers.classic.catboost import CatBoost
from csmt.classifiers.classic.deepforest import DeepForest
from csmt.classifiers.classic.hidden_markov_model import HMM
from csmt.classifiers.keras.lstm import LSTMKeras
from csmt.classifiers.torch.mlp import MLPTorch
from csmt.classifiers.torch.mlp import AlertNetTorch
from csmt.classifiers.torch.mlp import IdsNetTorch
from csmt.classifiers.torch.mlp import DeepNetTorch
from csmt.classifiers.torch.lr import LRTorch
from csmt.classifiers.torch.rnn import RNNTorch
from csmt.classifiers.torch.rnn import LSTMTorch
from csmt.classifiers.torch.tcn import TCNTorch
from csmt.classifiers.torch.transformer import TransformerTorch
from csmt.classifiers.torch.resnet import ResNetTorch
from csmt.classifiers.torch.ft_transformer import FTTransformerTorch
from csmt.classifiers.torch.cnn import CNNTorch,CNNMnistTorch,LeNetMnistTorch
from csmt.classifiers.keras.mlp import MLPKeras
from csmt.classifiers.keras.lstm import LSTMTextKeras
from csmt.classifiers.tensorflow.cnn import CNNTensor
from csmt.classifiers.tensorflow.mlp import MLPTensor
from csmt.classifiers.tensorflow.fsnet import FsnetTensor
from csmt.classifiers.anomaly_detection.KitNET import KitNET
from csmt.classifiers.anomaly_detection.IsolationForest import IsolationForest
from csmt.classifiers.anomaly_detection.other_anomaly import *
from csmt.classifiers.anomaly_detection.autoencoder import AbAutoEncoder
from csmt.classifiers.anomaly_detection.diff_rf import DIFFRF
from csmt.classifiers.ensemble.ensemble import SoftEnsembleModel
from csmt.classifiers.ensemble.ensemble import HardEnsembleModel
from csmt.classifiers.ensemble.ensemble import StackingEnsembleModel
from csmt.classifiers.ensemble.ensemble import BayesEnsembleModel
from csmt.classifiers.ensemble.ensemble import TransferEnsembleModel

from csmt.classifiers.model_op import ModelOperation
import numpy as np
from csmt.utils import get_number_labels

from csmt.estimators.classification.hard_ensemble import HardEnsemble

def model_dict(algorithm,input_size,output_size):

    models_dic={
        'lr':LogisticRegression,
        'knn':KNearestNeighbours,
        'dt':DecisionTree,
        'nb':NaiveBayes,
        'svm':SupportVectorMachine,
        'rf':RandomForest,
        'adaboost':Adaboost,
        'omni':Omni,
        'xgboost':XGBoost,
        'hmm':HMM,
        'lightgbm':LightGBM,
        'catboost':CatBoost,
        'deepforest':DeepForest,
        'mlp_torch':MLPTorch,
        'alertnet_torch':AlertNetTorch,
        'idsnet_torch':IdsNetTorch,
        'deepnet_torch':DeepNetTorch,
        'lr_torch':LRTorch,
        'rnn_torch':RNNTorch,
        'lstm_torch':LSTMTorch,
        'tcn_torch':TCNTorch,
        'resnet_torch':ResNetTorch,
        'transformer_torch':TransformerTorch,
        'cnn_tensor':CNNTensor,
        'mlp_tensor':MLPTensor,
        'fsnet_tensor':FsnetTensor,
        'ft_transformer':FTTransformerTorch,
        'cnn_torch':CNNTorch,
        'cnn_mnist_torch':CNNMnistTorch,
        'lenet_mnist_torch':LeNetMnistTorch,
        'mlp_keras':MLPKeras,
        'lstm_keras':LSTMKeras,
        'lstm_text_keras': LSTMTextKeras,
        'kitnet':KitNET,
        'ae':AbAutoEncoder,
        'if':IsolationForest,
        'ocsvm':AbOCSVM,
        'hbos':AbHBOS,
        'vae':AbVAE,
        'diff-rf':DIFFRF,
        'soft_ensemble':SoftEnsembleModel,
        'hard_ensemble':HardEnsembleModel,
        'stacking_ensemble':StackingEnsembleModel,
        'bayes_ensemble':BayesEnsembleModel
    }
    return models_dic[algorithm](input_size=input_size,output_size=output_size)

def get_model(models_name,input_size,output_size):
    """
    :param models_name: models name
    :param input_size:
    :param output_size: 
    :return: 
    """
    models_array=[]
    for i in range(len(models_name)):
        models_array.append(model_dict(models_name[i],input_size,output_size))
    return models_array,models_name
    

def models_train(datasets_name,models_name,X_train,y_train,X_val,y_val,if_adv=False):
    """
     models array training implementation.
    :param datasets_name: datasets name
    :param models_name: models name
    :param X_train:
    :param y_train:
    :param X_val: 
    :param y_val: 
    :return: 
    """
    input_size=X_train.shape[1]
    output_size=get_number_labels(y_train)
    original_models_array,algorithms_name=get_model(models_name,input_size,output_size)
    models_array=[]
    for i in range(0,len(original_models_array)):
        model_=ModelOperation()
        trained_model=model_.train(algorithms_name[i],datasets_name,original_models_array[i],X_train,y_train,X_val,y_val,if_adv)
        models_array.append(trained_model)

    return models_array

def models_load(datasets_name,models_name):
    """
     models array loading implementation.
    :param datasets_name: datasets name
    :param models_name: models name
    :return: 
    """
    models_array=[]
    for i in range(0,len(models_name)):
        model_=ModelOperation()
        trained_model=model_.load(datasets_name,models_name[i])
        models_array.append(trained_model)
    return models_array

def models_predict(models,models_name,X_test,y_test):
    """
     models array prediction implementation.
    :param models:
    :param X_test: 
    :param y_test: 
    :return: 
    """
    len_y=get_number_labels(y_test)
    y_pred_arr=np.zeros((len(models),X_test.shape[0],len_y))
    for i in range(len(models)):
        y_pred=models[i].predict(X_test)
        y_pred_arr[i]=y_pred

    return y_test,y_pred_arr

def models_predict_anomaly(models,models_name,X_test,y_test):
    """
     models array prediction implementation.
    :param models:
    :param X_test: 
    :param y_test: 
    :return: 
    """
    len_y=get_number_labels(y_test)
    y_pred_arr=np.zeros((len(models),X_test.shape[0],len_y))
    for i in range(len(models)):
        y_pred=models[i].predict_anomaly(X_test,y_test)
        y_pred_arr[i]=y_pred
    return y_test,y_pred_arr
    

