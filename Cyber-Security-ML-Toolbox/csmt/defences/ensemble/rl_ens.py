
import numpy as np
from csmt.config import CSMT_NUMPY_DTYPE
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.estimators.estimator import BaseEstimator
from csmt.classifiers.scores import get_class_scores
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from csmt.zoopt.gradient_free_optimizers  import HillClimbingOptimizer
from csmt.defences.ensemble.distribute import get_distribute

def RLEnsemble(models,X_test,y_test):

    output_size=len(np.unique(y_test))
    ens_w=1.0/len(models)*np.ones(len(models),dtype=CSMT_NUMPY_DTYPE)

    def get_ens_result(w):
        y_pred_all=np.zeros((X_test.shape[0],output_size))
        w_new=np.zeros(len(models))
        for i in range(len(models)):
            w_new[i]=w['x'+str(i)]
        w_new=get_distribute(w_new,len(models))
        for i in range(0,len(models)):
            y_pred = models[i].predict(X_test)
            y_pred_all=y_pred_all+y_pred*w_new[i]
        y_pred_all=np.argmax(y_pred_all, axis=1)
        result=get_class_scores(y_test, y_pred_all)
        goal=result[3]
        return goal

    ens_search_space={}
    for i in range(len(models)):
        ens_search_space.update({'x'+str(i):np.arange(0.01,0.99,0.01)})

    opt=HillClimbingOptimizer(ens_search_space,random_state=20)

    opt.search(get_ens_result, n_iter=100,verbosity=False)

    # print(opt.best_para)
    w_result=np.zeros(len(models))
    for i in range(len(models)):
        w_result[i]=opt.best_para['x'+str(i)]
    w_result=get_distribute(w_result,len(models))
    print(w_result)
    
    # history=opt.score_l
    # print(history)





