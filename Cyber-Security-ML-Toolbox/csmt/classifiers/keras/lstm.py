'''
Author: your name
Date: 2021-03-24 21:41:48
LastEditTime: 2021-07-12 09:39:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/torch/multi_layer_perceptron.py
'''
from collections import OrderedDict
from csmt.classifiers.abstract_model import AbstractModel


class LSTMTextKeras(AbstractModel):
    def __init__(self, input_size,output_size):
        from keras.models import Sequential
        from keras.layers import Dense, Embedding
        from keras.layers import LSTM
        from csmt.estimators.classification.keras import KerasClassifier
        import os
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        max_features = 20000
        maxlen = 80  # cut texts after this number of words (among top max_features most common words)
        batch_size = 32
        print('Build model...')
        model = Sequential()
        model.add(Embedding(max_features, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        self.classifier = KerasClassifier(model=model,clip_values=(0,1))
    

class LSTMKeras(AbstractModel):

    def __init__(self, input_size,output_size):
        from csmt.classifiers.abstract_model import AbstractModel

        import os
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import keras
        from keras.layers import Dense,LSTM, Activation,Input
        from keras.models import load_model
        from keras.models import Sequential, Model
        from csmt.estimators.classification.keras import KerasClassifier
        import numpy as np

        SEED_VALUE=1
        np.random.seed(SEED_VALUE)
        layers = [
            LSTM(32,return_sequences=True, stateful=True,batch_input_shape=(1,input_size,1)),
            LSTM(32, return_sequences=True, stateful=True),
            LSTM(32, stateful=True),
            Dense(output_size,activation='sigmoid')
        ]
        model = keras.Sequential()
        for layer in layers:
            model.add(layer)
            print(layer.output_shape)
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        self.classifier = KerasClassifier(model=model,clip_values=(0,1))



        

