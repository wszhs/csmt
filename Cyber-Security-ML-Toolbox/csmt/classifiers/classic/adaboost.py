'''
Author: your name
Date: 2021-03-24 19:36:20
LastEditTime: 2021-07-10 19:37:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/classic/random_forest.py
'''
from csmt.classifiers.abstract_model import AbstractModel
from sklearn.ensemble import AdaBoostClassifier
from csmt.estimators.classification.scikitlearn import SklearnClassifier

class Adaboost(AbstractModel):

    def __init__(self,
            input_size,
            output_size,
            learning_rate=0.1
        ):
        model = AdaBoostClassifier(learning_rate=learning_rate)
        self.classifier = SklearnClassifier(model=model,clip_values=(0,1))
