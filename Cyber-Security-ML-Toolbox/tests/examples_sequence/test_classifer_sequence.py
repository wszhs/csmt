
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load
import numpy as np
import torch
import matplotlib
import time
import random
import csmt.Interpretability.shap as shap

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

if __name__=='__main__':
     
     # print(X_train[0])
     # fft_values = np.fft.fft(X_train[0])
     # print(fft_values)
     
     arguments = sys.argv[1:]
     options = parse_arguments(arguments)
     datasets_name='flow_fsnet'
     orig_models_name=['fsnet_tensor']

     X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)
     
     print(X_train.shape)

     trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)
     # trained_models=models_load(datasets_name,orig_models_name)
     y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)
     
     # plot_headmap(X_test,y_pred[0][:,0],orig_models_name)

     table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
     
     # X_train = torch.tensor(X_train).float()
     # X_test = torch.tensor(X_test).float()
     # explainer = shap.DeepExplainer(trained_models[0].classifier.model, X_train)
     # shap_values = explainer.shap_values(X_test)
     
     # print(shap_values)

     
    
 









