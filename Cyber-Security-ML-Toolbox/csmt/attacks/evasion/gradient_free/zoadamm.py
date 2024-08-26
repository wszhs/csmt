
'''
Author: your name
Date: 2021-07-22 14:03:50
LastEditTime: 2021-07-27 14:44:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/zones.py
'''
import numpy as np
from typing import Optional, Union
import copy
from csmt.attacks.evasion.gradient_free.abstract_zero_order import AbstractZeroOrder
from csmt.config import CSMT_NUMPY_DTYPE
import logging

np.random.seed(10)

class ZOAdaMMMethod(AbstractZeroOrder):

    """
    Zo-adamm: Zeroth-order adaptive momentum method for black-box optimization
    | paper link: https://proceedings.neurips.cc/paper/2019/file/576d026223582a390cd323bef4bad026-Paper.pdf
    The adaptive momentum method (AdaMM) \cite{chen2019zo}, which uses past gradients to update descent directions and learning rates simultaneously, has become one of the most popular first-order optimization methods for solving machine learning problems.
    In this paper, we propose a zeroth-order AdaMM (ZO-AdaMM) algorithm, that generalizes AdaMM to the gradient-free regime.
    """

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        """
        :param q: number of random direction vectors
        :param mu: key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet
        :param eps_step: 
        :param init_const: regularization parameter prior to attack loss
        """
        sigma = 100
        grad_est=0
        d=x.shape[1]
        logging.debug(d)

        f_0,ignore=self.function_evaluation_cons(x,kappa,target_label,const,x)
        for i in range(q):
            u = np.random.normal(0, sigma, (1,d)).astype(CSMT_NUMPY_DTYPE)
            u_norm = np.linalg.norm(u)
            u = u/u_norm
            f_tmp, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,x)
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/(mu)

        return grad_est

    def generate_single(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs):
        """Generate a adversarial sample and return.
        :param x: A original input.
        :param y: Target value
        :return: A adversarial example.
        """
        c_mask = kwargs.get("c_mask")
        const=self.init_const
        x_adv=copy.deepcopy(x)
        x_adv_tmp=copy.deepcopy(x_adv)

        beta_1=0.9
        beta_2=0.99
        v_init = 1e-7 #0.00001

        # v_hat = v_init * np.ones((1,x.shape[1]))

        v = v_init * np.ones((1,x.shape[1]))
        delta_adv = np.zeros((1,self.max_iter,x.shape[1]),dtype=CSMT_NUMPY_DTYPE)
        x_adv_path=np.zeros((1,self.max_iter+1,x.shape[1]),dtype=CSMT_NUMPY_DTYPE)
        x_adv_path[0,0]=x
        m = np.zeros((1,x.shape[1]))
        
        for i in range(0,self.max_iter):
            # base_lr = self.eps_step/np.sqrt(i+1)
            base_lr = self.eps_step
            grad_est=self.gradient_estimation(self.mu,self.q,x_adv_tmp,self.kappa,y,const)

            if self.norm in [np.inf, "inf"]:
                m = beta_1 * m + (1-beta_1) * grad_est
                v = beta_2 * v + (1 - beta_2) * np.square(grad_est) ### vt
                # v_hat = np.maximum(v_hat,v)
                delta_adv[0,i] = delta_adv[0,i-1] - base_lr * m /np.sqrt(v)
                
                if c_mask is not None:
                    delta_adv[0,i] = np.where(c_mask == False, 0.0, delta_adv[0,i])
                delta_adv[0,i]=np.clip(delta_adv[0,i],-self.eps,self.eps)
            x_adv_tmp=x+delta_adv[0,i]
            x_adv_tmp=self._apply_feature_mapping(x_adv_tmp,**kwargs)
            x_adv_path[0,i+1]=x_adv_tmp
            
        x_adv = x_adv_tmp
        
        return x_adv,x_adv_path
