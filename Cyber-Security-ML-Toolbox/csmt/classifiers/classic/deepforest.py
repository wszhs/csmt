'''
Author: your name
Date: 2021-04-19 10:22:31
LastEditTime: 2021-07-10 20:17:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/classic/deepforest.py
'''
'''
Author: your name
Date: 2021-04-06 13:57:54
LastEditTime: 2021-04-16 14:09:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/classifiers/classic/xgboost.py
'''
from csmt.classifiers.abstract_model import AbstractModel

class DeepForest(AbstractModel):
    def __init__(self,input_size,output_size):
        from deepforest import CascadeForestClassifier
        from csmt.estimators.classification.ensemble_tree import EnsembleTree
        model=CascadeForestClassifier(random_state=1,verbose=0)
        self.classifier=EnsembleTree(model=model,nb_features=input_size, nb_classes=output_size,clip_values=(0,1))
