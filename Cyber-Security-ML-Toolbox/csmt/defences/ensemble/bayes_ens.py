
import numpy as np
from csmt.config import CSMT_NUMPY_DTYPE
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.estimators.estimator import BaseEstimator
from csmt.classifiers.scores import get_class_scores
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from csmt.defences.ensemble.distribute import get_distribute

def BayesEnsemble(models,X_test,y_test):

    output_size=len(np.unique(y_test))
    ens_w=1.0/len(models)*np.ones(len(models),dtype=CSMT_NUMPY_DTYPE)

    def get_ens_result(w):
        y_pred_all=np.zeros((X_test.shape[0],output_size))
        w_new=get_distribute(w,len(models))
        for i in range(0,len(models)):
            y_pred = models[i].predict(X_test)
            y_pred_all=y_pred_all+y_pred*w_new[i]
        y_pred_all=np.argmax(y_pred_all, axis=1)
        result=get_class_scores(y_test, y_pred_all)
        goal=result[3]
        return goal

    ens_bound=[]
    ens_keys=[]
    for i in range(len(models)):
        ens_bound.append([0.01,0.99])
        ens_keys.append('x'+str(i))

    bo = BayesianOptimization(f=get_ens_result,pbounds={'x':ens_bound},random_state=7)
    bo.maximize(init_points=10,n_iter=20,distribute=None)

    max_x=np.array([bo.max['params'][key] for key in ens_keys])
    ens_w=get_distribute(max_x,len(models))


    # y_pred_all=np.zeros((X_test.shape[0],output_size))
    # for i in range(0,len(models)):
    #     y_pred = models[i].predict(X_test)
    #     y_pred_all=y_pred_all+y_pred*ens_w[i]
    # y_pred_all=np.argmax(y_pred_all, axis=1)
    # result=get_class_scores(y_test, y_pred_all)
    # print(result[3])



