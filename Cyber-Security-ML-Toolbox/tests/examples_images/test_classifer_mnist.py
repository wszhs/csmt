
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load
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
     orig_models_name=['cnn_mnist_torch']
     datasets_name='mnist'

     X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)

     trained_models=models_train(datasets_name,orig_models_name,X_train,y_train,X_val,y_val)
         y_test,y_pred=models_predict(trained_models,orig_models_name,X_test,y_test)

     table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
     




     
    
 









