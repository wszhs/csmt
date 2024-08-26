'''
Author: your name
Date: 2021-04-01 17:38:22
LastEditTime: 2021-08-02 08:54:11
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/evasion_attack.py
'''
import numpy as np
from csmt.attacks.evasion.util import get_distribute
from csmt.config import CSMT_NUMPY_DTYPE
from csmt.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from csmt.get_model_data import models_predict
from csmt.attacks.evasion.gradient_attack import BIMAttack,FGSMAttack,PGDAttack,CWAttack,JSMAAttack,DeepFoolAttack,UniversalAttack
from csmt.attacks.evasion.gradient_free_attack import BoundAttack, HSJAttack, ZOOAttack, DEAttack,GAAttack,\
     ZOSGDAttack,ZONESAttack,ZOSCDAttack,ZOAdaMMAttack,BayesAttack,GradFreeAttack,OpenboxAttack,MimicryAttack,GANAttack

def evasion_dict(model,algorithm,estimator_features=None):
    evasion_dic={
        'fgsm':FGSMAttack,
        'fgsm_l1':FGSMAttack,
        'fgsm_l2':FGSMAttack,
        'universal':UniversalAttack,
        'pgd':PGDAttack,
        'pgd_l1':PGDAttack,
        'pgd_l2':PGDAttack,
        'cw':CWAttack,
        'bim':BIMAttack,
        'jsma':JSMAAttack,
        'deepfool':DeepFoolAttack,
        'zoo':ZOOAttack,
        'hsj':HSJAttack,
        'bound':BoundAttack,
        'zones':ZONESAttack,
        'zosgd':ZOSGDAttack,
        'zoscd':ZOSCDAttack,
        'gan':GANAttack,
        'zoadamm':ZOAdaMMAttack,
        'de':DEAttack,
        'ga':GAAttack,
        'bayes':BayesAttack,
        'grad_free':GradFreeAttack,
        'openbox':OpenboxAttack,
        'mimicry':MimicryAttack
    }

    if 'l1' in algorithm:
        return evasion_dic[algorithm](estimator=model,eps=1.5,eps_step=1.5,max_iter=1,norm=1,estimator_features=estimator_features)
    elif 'l2' in algorithm:
        return evasion_dic[algorithm](estimator=model,eps=0.6,eps_step=0.6,max_iter=1,norm=2,estimator_features=estimator_features)
    else:
        return evasion_dic[algorithm](estimator=model,eps=0.3,eps_step=0.3,max_iter=1,norm=np.inf,estimator_features=estimator_features)

def EvasionAttack(attack_models,evasion_algorithm,X_test,y_test,constraints, estimator_features=None):
    
    if estimator_features is None:
        estimator_features=np.arange(X_test.shape[1])
        
    attack=evasion_dict(attack_models[0],evasion_algorithm[0],estimator_features)

    # c_mask=constraints['c_mask']
    # c_range=constraints['c_range']
    # c_eq=constraints['c_eq']
    # c_ueq=constraints['c_ueq']

    X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)

    return X_adv,y_adv,X_adv_path


def TransferEnsembleEvasionAttack(attack_models,evasion_algorithm,X_test,y_test,constraints,estimator_features=None):
    
    if estimator_features is None:
        estimator_features=np.arange(X_test.shape[1])
        
    transfer_weight=1.0/len(attack_models)*np.ones(len(attack_models),dtype=CSMT_NUMPY_DTYPE)
    print(transfer_weight)
    # print("开始迁移集成攻击")
    X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]),dtype=CSMT_NUMPY_DTYPE)
    for i in range(len(attack_models)):
        attack=evasion_dict(attack_models[i],evasion_algorithm[0],estimator_features)
        X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
        X_adv_all+=X_adv*transfer_weight[i]
    return X_adv_all,y_adv,X_adv_path


def TransferBayesEnsembleEvasionAttack(attack_models,attack_model_name,evaluation_models,evaluation_models_name,evasion_algorithm,X_test,y_test,constraints,estimator_features=None):
    print("开始贝叶斯迁移集成攻击")
    transfer_weight=1.0/len(attack_model_name)*np.ones(len(attack_model_name),dtype=CSMT_NUMPY_DTYPE) 
    print(transfer_weight)
    
    def get_result(w):
        X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]),dtype=CSMT_NUMPY_DTYPE)
        y_pred_all=np.zeros((X_test.shape[0],2))
        w_new=get_distribute(w)
        for i in range(len(attack_model_name)):
            attack=evasion_dict(attack_models[i],evasion_algorithm[0],estimator_features)
            X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
            X_adv_all+=X_adv*w_new[i]

        y_test_adv,y_pred_all=models_predict(evaluation_models,evaluation_models_name,X_adv_all,y_test)

        #扩展到全部攻击成功率
        y_test_1=y_test
        y_pred_arr_1=y_pred_all

        K=len(evaluation_models)
        adv_maps = np.full((K,len(y_test_1)), False)
        for k in range(K):
            y_pred=np.argmax(y_pred_arr_1[k], axis=1)
            adv_maps[k]=(y_pred != y_test_1)
        asr_all = np.full(len(y_test_1), True)
        for adv_map in adv_maps:
            asr_all = np.logical_and(adv_map, asr_all)
        print ('zhs_ASR_all: %.2f %%' % (100 * np.sum(asr_all) / float(len(y_test_1))))
        goal=(np.sum(asr_all) / float(len(y_test_1)))
        return goal

    bound=[]
    keys=[]
    for i in range(len(attack_model_name)):
        bound.append([0.01,0.99])
        keys.append('x'+str(i))

    bo = BayesianOptimization(f=get_result,pbounds={'x':bound},random_state=7)
    
    bo.maximize(init_points=10,n_iter=20,distribute=None)
    print(bo.max['params'])
    max_x=np.array([bo.max['params'][key] for key in keys])
    weight_distribute=get_distribute(max_x)
    print(weight_distribute)  
    transfer_weight=weight_distribute

    X_adv_all=np.zeros((X_test.shape[0],X_test.shape[1]))
    for i in range(len(attack_model_name)):
        attack=evasion_dict(attack_models[i],evasion_algorithm[0],estimator_features)
        X_adv,y_adv,X_adv_path=attack.generate(X_test,y_test)
        X_adv_all+=X_adv*transfer_weight[i]
    return X_adv_all,y_adv,X_adv_path
