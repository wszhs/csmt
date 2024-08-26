
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
from csmt.defences.ensemble.bayes_ens import BayesEnsemble
from csmt.defences.ensemble.rl_ens import RLEnsemble
from csmt.defences.ensemble.nash_ens import NashEnsemble
from csmt.defences.ensemble.nashrl_ens import NashRLEnsemble
import numpy as np
import torch
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

if __name__=='__main__':
     
     arguments = sys.argv[1:]
     options = parse_arguments(arguments)
     datasets_name=options.datasets
     orig_models_name=options.algorithms
     attack_model_name=options.attack_models
     evasion_algorithm=options.evasion_algorithm

     X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
     X_train,y_train,X_test,y_test=X_train,y_train,X_test,y_test

     X_test,y_test=X_test[0:150],y_test[0:150]

     X_test_1=X_test[y_test==1]
     X_test_0=X_test[y_test==0]

     setup_seed(20)
     attack_models=models_train(datasets_name,attack_model_name,X_train,y_train,X_val,y_val)
     setup_seed(20)
     trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)

     # BayesEnsemble(trained_models,X_test,y_test)

     NashEnsemble(trained_models,attack_models,attack_model_name,evasion_algorithm,X_test,y_test)



     
    
 









