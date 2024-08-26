
import numpy as np
import copy
import sys
from csmt.config import CSMT_NUMPY_DTYPE
from csmt.get_model_data import get_datasets,models_train,parse_arguments,models_train,print_results,models_predict

np.random.seed(10)
class MimicryMethod():
    def __init__(self,estimator=None,norm=np.inf,eps=0.2,eps_step=0.01,max_iter=20):
        self.estimator=estimator
        self.norm=norm
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.mimic_x=None
        self.orig_prob=None
        
    def get_mimic_set(self):
        arguments = sys.argv[1:]
        options = parse_arguments(arguments)
        datasets_name=options.datasets
        attack_model_name=options.attack_models
        orig_models_name=options.algorithms
        evasion_algorithm=options.evasion_algorithm
        X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)
        X_train_0=X_train[y_train==0]
        y_train_0=y_train[y_train==0]
        return X_train_0
        

    def generate(self, X, y):
        x_adv_path=np.zeros((1,2,X.shape[0]))
        self.mimic_x=self.get_mimic_set()
        self.orig_prob = self.estimator.predict(self.mimic_x)
        X_adv=copy.deepcopy(X)
        for i in range(X.shape[0]):
            if i%10==0:
                print(i)
            x_singe=X[i:i+1]
            y_single=y[i:i+1]
            x_adv_single=self.generate_single(x_singe,y_single)
            X_adv[i]=x_adv_single
        X_test_adv = np.append(X, X_adv, axis=0)
        y_test_adv = np.append(y, y, axis=0)
        return X_test_adv,x_adv_path

    def generate_single(self,x,y):

        x_orig=copy.deepcopy(x)
        # print(self.estimator.predict(x_orig)[0,y])
        x_adv=copy.deepcopy(x)
        delta_adv = np.zeros((self.max_iter,1,x.shape[1])).astype(CSMT_NUMPY_DTYPE)
        for i in range(0,self.max_iter):
            grad_est=self.gradient_estimation_mimicry(x,y)
            if self.norm in [np.inf, "inf"]:
                delta_adv[i] =delta_adv[i-1]-self.eps_step*grad_est
                delta_adv[i]=np.clip(delta_adv[i],-self.eps,self.eps)
            x_adv_tmp=copy.deepcopy(x_orig)
            x_adv_tmp=x_orig+delta_adv[i]
            print(self.estimator.predict(x_adv_tmp)[0,y])
        x_adv = x_adv+delta_adv[self.max_iter-1]
        return x_adv

    def gradient_estimation_mimicry(self,x,y):
        min_anomaly=np.inf
        min_dis=np.inf
        min_x_dis=None
        # print(self.mimic_x)
        # print(np.mean(self.mimic_x,axis=0))
        # grad_est=x-np.mean(self.mimic_x,axis=0)

        # for i in range(self.mimic_x.shape[0]):
        #     anomaly_score=self.orig_prob[i][1]
        #     if anomaly_score<min_anomaly:
        #         min_anomaly=anomaly_score
        #         min_x_dis=x-self.mimic_x[i]
        for i in range(self.mimic_x.shape[0]):
            x_dis=x-self.mimic_x[i]
            dis_=np.linalg.norm(x_dis,ord=2)
            if dis_<min_dis:
                min_dis=dis_
                min_x_dis=x_dis
        grad_est=min_x_dis
        return grad_est

