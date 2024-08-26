
import numpy as np
from csmt.config import CSMT_NUMPY_DTYPE
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.estimators.estimator import BaseEstimator
from csmt.classifiers.scores import get_class_scores
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from csmt.defences.ensemble.distribute import get_distribute
from csmt.attacks.evasion.evasion_attack import TransferEnsembleEvasionAttack

# TransferEnsembleEvasionAttack(attack_models,transfer_weight,attack_model_name,evasion_algorithm,X_test,y_test,upper=1,lower=0,feature_importance=None,mask=None)

def NashEnsemble(models,attack_models,attack_model_name,evasion_algorithm,X_test,y_test):

    output_size=len(np.unique(y_test))
    ens_w=1.0/len(models)*np.ones(len(models),dtype=CSMT_NUMPY_DTYPE)
    attack_w=1.0/len(attack_models)*np.ones(len(attack_models),dtype=CSMT_NUMPY_DTYPE)

    X_test_1=X_test[y_test==1]
    y_test_1=y_test[y_test==1]

    X_test_0=X_test[y_test==0]
    y_test_0=y_test[y_test==0]

    def get_result(w,X_test_adv,y_test_adv):
        y_pred_all=np.zeros((X_test_adv.shape[0],output_size))
        w_new=get_distribute(w,len(models))
        for i in range(0,len(models)):
            y_pred = models[i].predict(X_test_adv)
            y_pred_all=y_pred_all+y_pred*w_new[i]
        y_pred_all=np.argmax(y_pred_all, axis=1)
        result=get_class_scores(y_test_adv, y_pred_all)
        goal=result[3]
        return goal

    def get_attack_success(w):
        attack_w=get_distribute(w,len(attack_models))
        X_adv,y_adv,X_adv_path=TransferEnsembleEvasionAttack(attack_models,attack_w,attack_model_name,evasion_algorithm,X_test_1,y_test_1,upper=1,lower=0,feature_importance=None,mask=None)
        X_test_adv = np.append(X_test_0, X_adv, axis=0)
        y_test_adv = np.append(y_test_0, y_adv, axis=0)
        goal=get_result(ens_w,X_test_adv,y_test_adv)
        return 1-goal

    def get_ens_success(w):
        ens_w=get_distribute(w,len(models))
        X_adv,y_adv,X_adv_path=TransferEnsembleEvasionAttack(attack_models,attack_w,attack_model_name,evasion_algorithm,X_test_1,y_test_1,upper=1,lower=0,feature_importance=None,mask=None)
        X_test_adv = np.append(X_test_0, X_adv, axis=0)
        y_test_adv = np.append(y_test_0, y_adv, axis=0)
        goal=get_result(ens_w,X_test_adv,y_test_adv)
        return goal

    attack_bound=[]
    attack_keys=[]

    ens_bound=[]
    ens_keys=[]

    for i in range(len(attack_models)):
        attack_bound.append([0.01,0.99])
        attack_keys.append('x'+str(i))
    
    for i in range(len(models)):
        ens_bound.append([0.01,0.99])
        ens_keys.append('x'+str(i))


    for i in range(50):
        bo_attack = BayesianOptimization(f=get_attack_success,pbounds={'x':attack_bound},random_state=7)
        bo_ens = BayesianOptimization(f=get_ens_success,pbounds={'x':ens_bound},random_state=7)

        print('Attack!')
        bo_attack.maximize(init_points=5,n_iter=5)
        max_x=np.array([bo_attack.max['params'][key] for key in attack_keys])
        attack_w=get_distribute(max_x,len(attack_models))
        print(attack_w)

        print('Defense')
        bo_ens.maximize(init_points=5,n_iter=5)
        max_x=np.array([bo_ens.max['params'][key] for key in ens_keys])
        ens_w=get_distribute(max_x,len(models))
        print(ens_w)





