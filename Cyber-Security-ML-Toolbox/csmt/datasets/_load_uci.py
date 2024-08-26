from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
import pickle
import pandas as pd
import numpy as np
from os import path
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict
from scipy.io import loadmat

def load_uci():
    X,y=datasets.fetch_kddcup99(return_X_y=True)
    print(X)
    print(y)
