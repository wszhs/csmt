
from csmt.classifiers.abstract_model import AbstractModel

class MLPKeras(AbstractModel):
    """
    Multi-layer perceptron.
    """
    def __init__(self, input_size,output_size):
        
        import os
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import keras
        from keras.layers import Dense, Activation,Input,Dropout
        from keras.models import load_model
        from keras.models import Sequential, Model
        from csmt.estimators.classification.keras import KerasClassifier
        import numpy as np

        SEED_VALUE=1
        # tf.random.set_seed(SEED_VALUE)
        np.random.seed(SEED_VALUE)

        layers = [
            Dense(256, activation='relu',input_shape=(input_size,)),
            # Dropout(0.5),
            Dense(64,activation='relu'),
            # Dropout(0.5),
            Dense(32,activation='relu'),
            # Dropout(0.5),
            Dense(16,activation='relu'),
            # Dropout(0.5),
            Dense(output_size,activation='sigmoid')
        ]
        model = keras.Sequential()
        for layer in layers:
            model.add(layer)
        model.summary()
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        self.classifier = KerasClassifier(model=model,clip_values=(0,1))



        

