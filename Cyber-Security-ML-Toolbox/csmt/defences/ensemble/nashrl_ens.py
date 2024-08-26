
import numpy as np
from csmt.config import CSMT_NUMPY_DTYPE
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.estimators.estimator import BaseEstimator
from csmt.classifiers.scores import get_class_scores
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from csmt.defences.ensemble.distribute import get_distribute
from csmt.attacks.evasion.evasion_attack import TransferEnsembleEvasionAttack
from csmt.zoopt.gradient_free_optimizers  import HillClimbingOptimizer

def NashRLEnsemble(models,attack_models,attack_model_name,evasion_algorithm,X_test,y_test):

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
        w_new=np.zeros(len(attack_models))
        for i in range(len(attack_models)):
            w_new[i]=w['x'+str(i)]
        attack_w=get_distribute(w_new,len(attack_models))
        X_adv,y_adv,X_adv_path=TransferEnsembleEvasionAttack(attack_models,attack_w,attack_model_name,evasion_algorithm,X_test_1,y_test_1,upper=1,lower=0,feature_importance=None,mask=None)
        X_test_adv = np.append(X_test_0, X_adv, axis=0)
        y_test_adv = np.append(y_test_0, y_adv, axis=0)
        goal=get_result(ens_w,X_test_adv,y_test_adv)
        return 1-goal

    def get_ens_success(w):
        w_new=np.zeros(len(models))
        for i in range(len(models)):
            w_new[i]=w['x'+str(i)]
        ens_w=get_distribute(w_new,len(models))
        X_adv,y_adv,X_adv_path=TransferEnsembleEvasionAttack(attack_models,attack_w,attack_model_name,evasion_algorithm,X_test_1,y_test_1,upper=1,lower=0,feature_importance=None,mask=None)
        X_test_adv = np.append(X_test_0, X_adv, axis=0)
        y_test_adv = np.append(y_test_0, y_adv, axis=0)
        goal=get_result(ens_w,X_test_adv,y_test_adv)
        return goal

    ens_search_space={}
    for i in range(len(models)):
        ens_search_space.update({'x'+str(i):np.arange(0.01,0.99,0.01)})

    attack_search_space={}
    for i in range(len(attack_models)):
        attack_search_space.update({'x'+str(i):np.arange(0.01,0.99,0.01)})


    for i in range(50):

        opt_attack=HillClimbingOptimizer(attack_search_space,random_state=20)
        opt_ens=HillClimbingOptimizer(ens_search_space,random_state=20)

        print('Attack!')
        opt_attack.search(get_attack_success, n_iter=10,verbosity=False)
        attack_w=np.zeros(len(attack_models))
        for i in range(len(attack_models)):
            attack_w[i]=opt_attack.best_para['x'+str(i)]
        attack_w=get_distribute(attack_w,len(attack_models))
        attack_history=opt_attack.score_l
        print(np.max(attack_history))
        print(attack_history)
        # print(attack_w)

        print('Defense')
        opt_ens.search(get_ens_success, n_iter=10,verbosity=False)
        ens_w=np.zeros(len(models))
        for i in range(len(models)):
            ens_w[i]=opt_ens.best_para['x'+str(i)]
        ens_w=get_distribute(ens_w,len(models))
        # print(ens_w)
        ens_history=opt_ens.score_l
        print(np.max(ens_history))
        print(ens_history)




