
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
import numpy as np
from csmt.get_model_data import get_datasets,models_train,parse_arguments,models_train,print_results,models_predict,get_results
from csmt.attacks.evasion.evasion_attack import EvasionAttack
import torch
import random
from csmt.utils import get_logger
from csmt.figure.visualml.plot_importance import plot_vec

    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
# 设置随机数种子
setup_seed(20)
logger=get_logger()

def Sensitive(attack_models,evasion_algorithm,trained_models,orig_models_name,X_test_0,y_test_0,X_test_1,y_test_1,constraints,estimator_features):
    X_adv,y_adv,X_adv_path=EvasionAttack(attack_models,evasion_algorithm,X_test_1,y_test_1,constraints,estimator_features) 
    X_test_adv = np.append(X_test_0, X_adv, axis=0)
    y_test_adv = np.append(y_test_0, y_adv, axis=0)
    y_test_adv,y_pred_adv=models_predict(trained_models,orig_models_name,X_test_adv,y_test_adv)
    results=get_results(datasets_name,orig_models_name,y_test_adv,y_pred_adv)
    print(results[0][-1])
    return results[0][-1]

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    attack_model_name=options.attack_models
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm
    
    orig_models_name_arr=[['lr'],['rf'],['xgboost'],['mlp_torch'],['rnn_torch'],['lstm_torch'],['alertnet_torch'],['idsnet_torch'],['deepnet_torch']]
    # orig_models_name_arr=[['idsnet_torch'],['deepnet_torch'],['alertnet_torch']]
    
    for i in range(len(orig_models_name_arr)):
          orig_models_name=orig_models_name_arr[i]
          print(orig_models_name)
          X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)

          setup_seed(20)
          print(orig_models_name)
          trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)
          attack_models=trained_models
           
          y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)

          table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

          X_test,y_test,y_pred=X_test[0:1000],y_test[0:1000],y_pred[0:1000]

          X_test_1=X_test[y_test==1]
          y_test_1=y_test[y_test==1]
          X_test_0=X_test[y_test==0]
          y_test_0=y_test[y_test==0]
          print(X_test_1.shape)
          print(X_test_0.shape)

          Sensitive_arr=[]
          for i in range(X_test.shape[1]):
               estimator_features=np.array([i])
               sensi=Sensitive(attack_models,evasion_algorithm,trained_models,orig_models_name,X_test_0,y_test_0,X_test_1,y_test_1,constraints,estimator_features)
               Sensitive_arr.append(sensi)
          Sensitive_arr=np.array(Sensitive_arr)
          print(Sensitive_arr)
          plot_vec(Sensitive_arr,'/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/sensitive/kitsune/scan/'+datasets_name+orig_models_name[0])

     



