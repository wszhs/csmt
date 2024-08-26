'''
Author: your name
Date: 2021-04-01 17:45:01
LastEditTime: 2021-06-07 17:18:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/de.py
'''
from csmt.zoopt.DE import DE
from tqdm import tqdm
import numpy as np
from typing import Optional, Union
from csmt.config import CSMT_NUMPY_DTYPE
    
class DEEvasionAttack():

    @staticmethod
    def get_score(p):
        p=p.astype(CSMT_NUMPY_DTYPE)
        DEEvasionAttack.count=DEEvasionAttack.count+1
        score=DEEvasionAttack.estimator.predict(p.reshape(1,-1))
        return -score[0][0]

    def __init__(
        self,
        estimator,
        eps:Union[int, float, np.ndarray] = 0.3,
        eps_step:Union[int, float, np.ndarray] = 0.1,
        max_iter:int = 100,
        norm: Union[int, float, str] = np.inf,
    )-> None:
        """
        Create a :class:`AbstractZeroOrder` instance.
        :param estimator: A trained classifier
        :param norm:  The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps:Attack step size (input variation).
        :param eps_step: Step size of input variation for minimal perturbation computation.
        :param init_const: regularization parameter prior to attack loss
        """

        DEEvasionAttack.estimator=estimator
        self.norm=norm
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.count=0

    def generate(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs):
        """Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs.
        :param y: Target values
        :return: An array holding the adversarial examples, the adversarial example generation path
        """
        X_adv_path=np.zeros((X.shape[0],2,X.shape[1]))

        X_size=X.shape[1]
        num=X.shape[0]
        X_adv=np.zeros(shape=(num,X_size)).astype(CSMT_NUMPY_DTYPE)

        for i in tqdm(range(num)):
            DEEvasionAttack.count=0
            x,x_adv_path=self._get_single_x(X[i],X_size)
            X_adv[i]=x
            X_adv_path[i]=x_adv_path

        return X_adv,X_adv_path
        
    def _get_single_x(self,x,chrom_length):
                
        """Generate a adversarial sample and return.
        :param x: A original input.
        :param chrom_length: 
        :return: A adversarial example.
        """

        bound=np.zeros((x.shape[0],2),dtype=float)
        x_adv_path=np.zeros((1,2,x.shape[0]))
        x_adv_path[0,0]=x

        for i in range(x.shape[0]):
            bound[i]=np.array([-self.eps,self.eps])+x[i]
            bound=np.clip(bound, 0, 1)
        lb=bound[:,0]
        ub=bound[:,1]

        de = DE(func=DEEvasionAttack.get_score, n_dim=chrom_length, size_pop=10, max_iter=50, lb=lb, ub=ub)
        best_x, best_y = de.run()
        x_adv_path[0,1]=best_x

        return best_x,x_adv_path

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
            




