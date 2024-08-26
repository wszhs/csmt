
import numpy as np
from csmt.attacks.evasion.gradient_free.abstract_zero_order import AbstractZeroOrder
from csmt.config import CSMT_NUMPY_DTYPE
import logging

np.random.seed(10)
class ZOSCDMethod(AbstractZeroOrder):

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        """
        :param q: number of random direction vectors
        :param mu: key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet
        :param eps_step: 
        :param init_const: regularization parameter prior to attack loss
        A simplified version of ZOO

        ZOSCD uses hinge loss in Equation:
        $$
            \max \left(\arg _{i \neq t} \max (\log (f(x+r, i)))-\log (f(x+r, t)),-k\right)
        $$
        where the input $x$, correctly classified by the classifier $f$ , is perturbed with $r$, such that the resulting adversarial example is $x+r$.
        ZOSCD uses the symmetric difference quotient to estimate the gradient and Hessian:
        $$
            \begin{array}{c}
            \frac{\partial f(x)}{\partial x_{i}} \approx \frac{f\left(x+h * e_{i}\right)-f\left(x-h * e_{i}\right)}{2 h} \\
            \frac{\partial^{2} f(x)}{\partial x_{i}^{2}} \approx \frac{f\left(x+h * e_{i}\right)-2 f(x)+f\left(x-h * e_{i}\right)}{h^{2}}
            \end{array}
        $$
        where $e_i$ denotes the standard basis vector with the i-th component as 1, and $h$ is a small constant.

        """
        grad_est=0
        d=x.shape[1]
        idx_coords_random=np.random.randint(d,size=q)

        for id_coord in range(q):
            idx_coord=idx_coords_random[id_coord]
            u = np.zeros(d).astype(CSMT_NUMPY_DTYPE)
            u[idx_coord]=1
            u=np.resize(u,x.shape)
            logging.debug(u)
            f_old, ignore = self.function_evaluation_cons(x-mu*u,kappa,target_label,const,x)
            f_new, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,x)
            grad_est=grad_est+ (d/q)*u*(f_new-f_old)/(2*mu)

        return grad_est

