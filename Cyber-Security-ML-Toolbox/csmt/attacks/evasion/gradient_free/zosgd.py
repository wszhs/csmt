
import numpy as np
from csmt.attacks.evasion.gradient_free.abstract_zero_order import AbstractZeroOrder
from csmt.config import CSMT_NUMPY_DTYPE
import logging

np.random.seed(10)
class ZOSGDMethod(AbstractZeroOrder):

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        """
        :param q: number of random direction vectors
        :param mu: key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet
        :param eps_step: 
        :param init_const: regularization parameter prior to attack loss

        Ghadimi et al. \cite{ghadimi2013stochastic} aim to estimate the gradient $\triangledown_\mathbf{x}g(\mathbf{x})$ of the ML model $f'$ more accurately to generate adversarial feature $\hat{\mathbf{x}}$.
        ZOSGD gradient estimation is constructed by forward comparing two function values at random directions: 
        $\nabla_\mathbf{x}g(\mathbf{x})=(1/ \sigma)[g(\mathbf{x}+\sigma\mathbf{u})-g(\mathbf{x})] \mathbf{u}$, 
        where $\mathbf{u}$ is a random vector drawn uniformly from the sphere of a unit ball, and $\sigma> 0$ is a small step size, known as the smoothing parameter. 
        The random direction vector $\mathbf{u}$ is drawn from the standard Gaussian distribution.
        We evaluate the gradient with a population of $q$ points sampled under this scheme:
        $$
        \nabla \mathbb{E}[g(\mathbf{x})] \approx \frac{1}{\sigma q} \sum_{i=1}^{q} \mathbf{u}_{i} [g\left(\mathbf{x}+\sigma \mathbf{u}_{i}\right)-g(\mathbf{x})].
        $$
        """
        sigma = 100
        grad_est=0
        d=x.shape[1]
        logging.debug(d)
        
        # print(self.estimator_features)
        if self.estimator_features is None:
            len_feas=d
        else:
            len_feas=self.estimator_features.shape[0]

        f_0,ignore=self.function_evaluation_cons(x,kappa,target_label,const,x)
        
        for i in range(q):
            
            u_feas = np.random.normal(0, sigma, (1,len_feas)).astype(CSMT_NUMPY_DTYPE)
            u_norm_feas = np.linalg.norm(u_feas)
            u_feas = u_feas/u_norm_feas
            u=np.zeros(d).astype(CSMT_NUMPY_DTYPE)
            u[self.estimator_features]=u_feas
            
            f_tmp, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,x)
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/(mu)
        return grad_est
