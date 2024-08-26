'''
Author: your name
Date: 2021-04-01 17:30:49
LastEditTime: 2021-08-04 09:16:22
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/abstract_evasion.py
'''
from abc import abstractmethod
import copy
import logging
import numpy as np
from typing import Optional, Union
from csmt.config import CSMT_NUMPY_DTYPE
from csmt.attacks.attack import EvasionAttack
from csmt.utils import get_logger
from tqdm import tqdm

class AbstractZeroOrder(EvasionAttack):

    def __init__(
        self,
        estimator,
        norm: Union[int, float, str] = np.inf,
        eps:Union[int, float, np.ndarray] = 0.3,
        eps_step:Union[int, float, np.ndarray] = 0.1,
        max_iter:int = 100,
        q: int=50,
        mu: float=0.2,
        kappa: float=1e-10,
        init_const: int=1,
        estimator_features: np.array=None
    )-> None:
        """
        Create a :class:`AbstractZeroOrder` instance.
        :param estimator: A trained classifier
        :param norm:  The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps:Attack step size (input variation).
        :param q: number of random direction vectors
        :param mu: key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet
        :param eps_step: Step size of input variation for minimal perturbation computation.
        :param init_const: regularization parameter prior to attack loss
        """
        super().__init__(estimator=estimator)
        self.norm=norm
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.q = q 
        self.mu = mu
        self.kappa = kappa
        self.init_const = init_const
        self.estimator_features=estimator_features
        AbstractZeroOrder._check_params(self)


    def generate(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs):

        """Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs.
        :param y: Target values
        :return: An array holding the adversarial examples, the adversarial example generation path
        """
        # Check that `y` is provided for targeted attacks
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")
        X_adv=copy.deepcopy(X)
        X_adv_path=np.zeros((X_adv.shape[0],self.max_iter+1,X_adv.shape[1]))
        for i in tqdm(range(X.shape[0])):
            if i%10==0:
                logging.debug(i)
            x_singe=X[i:i+1]
            y_single=y[i:i+1]
            x_adv_single,x_adv_path=self._generate_single(x_singe,y_single,**kwargs)
            X_adv[i]=x_adv_single
            X_adv_path[i]=x_adv_path

        return X_adv,X_adv_path

    def _generate_single(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs):
        
        """Generate a adversarial sample and return.
        :param x: A original input.
        :param y: Target value
        :return: A adversarial example.
        """
        c_mask = kwargs.get("c_mask")

        const=self.init_const

        x_orig=copy.deepcopy(x)
        x_adv=copy.deepcopy(x)
        x_adv_tmp=copy.deepcopy(x_orig)
        
        delta_adv = np.zeros((1,self.max_iter,x.shape[1]),dtype=CSMT_NUMPY_DTYPE)
        x_adv_path=np.zeros((1,self.max_iter+1,x.shape[1]),dtype=CSMT_NUMPY_DTYPE)

        x_adv_path[0,0]=x
        iter=self.max_iter-1

        for i in range(0,self.max_iter):

            base_lr = self.eps_step
            grad_est=self.gradient_estimation(self.mu,self.q,x_adv_tmp,self.kappa,y,const)
            if self.norm in [np.inf, "inf"]:
                delta_adv[0,i] =delta_adv[0,i-1]-base_lr*np.sign(grad_est)

                if c_mask is not None:
                    delta_adv[0,i] = np.where(c_mask == False, 0.0, delta_adv[0,i])

                delta_adv[0,i]=np.clip(delta_adv[0,i],-self.eps,self.eps)

            x_adv_tmp=x_orig+delta_adv[0,i]
            x_adv_tmp=self._apply_feature_mapping(x_adv_tmp,**kwargs)
            x_adv_path[0,i+1]=x_adv_tmp

            logging.debug(self.estimator.predict(x_adv_tmp)[0,y])
            if self.estimator.predict(x_adv_tmp)[0,y]<self.estimator.predict(x_adv_tmp)[0,0]:
                iter=i
                break

        x_adv = x_adv_tmp
        return x_adv,x_adv_path

    def function_evaluation_cons(self, x_adv:np.ndarray, kappa, target_label, const,x:np.ndarray):
        
        """ Objection Function Evaluation for constrained optimization formulation.
        :param x: A original input.
        :param x_adv: A adversarial input.
        :return: Loss.
        """
        orig_prob = self.estimator.predict(x_adv)
        tmp = orig_prob.copy()
        tmp[0, target_label] = 0
        Loss1=orig_prob[0, target_label] 
        Loss2 = np.linalg.norm(x_adv - x) ** 2 
        return Loss1, Loss2

    def _apply_feature_mapping(self,x,**kwargs):
        """
        We have four constraints
        c_mask: constraint_mask means restrict feature modification
        c_range: constraint_range indicates the upper and lower constraints for modification
        c_eq: constraint_eq indicates equality constraints
        c_ueq: constraint_ueq indicates an inequality constraint
        """
        # four constraints
        c_mask = kwargs.get("c_mask")
        c_range = kwargs.get("c_range")
        c_eq = kwargs.get("c_eq")
        c_ueq = kwargs.get("c_ueq")
        """
        limits the scope of each feature
        """

        if c_range is not None:
            x = np.clip(x, c_range[0], c_range[1])
        """
        Implement equality and inequality constraints
        if c_eq is not None:
            penalty_eq = ...
        if c_ueq is not None:
            penalty_ueq = ...

        """
        return x

    def _invert_feature_mapping(self,x):
        """
        From feature vectors to problem vectors
        """
        raise NotImplementedError("This function is about to be implemented")

    @abstractmethod
    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        """ Abstract interface for gradient evaluation.
        """
        raise NotImplementedError("This method is abstract, you should implement it somewhere else!")

    def _check_params(self):
        if not isinstance(self.q, (int, np.int)):
            raise ValueError("The number of 1must be an integer greater than zero.")