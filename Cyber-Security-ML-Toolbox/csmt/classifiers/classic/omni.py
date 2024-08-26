'''
Author: your name
Date: 2021-03-24 19:36:20
LastEditTime: 2021-07-10 19:37:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/classic/random_forest.py
'''
from pyexpat import model
from csmt.classifiers.abstract_model import AbstractModel
from csmt.estimators.classification.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

class Omni(AbstractModel):
    def __init__(self,
            input_size,
            output_size,
            learning_rate=0.1
        ):
        clf1 = LogisticRegression(random_state=1)
        clf2 = RandomForestClassifier(random_state=1)
        clf3 = GaussianNB()
        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
        params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}
        model = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
        self.classifier = SklearnClassifier(model=model,clip_values=(0,1))
