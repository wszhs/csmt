
import logging
import numpy as np
from csmt.attacks.evasion.gradient_free.abstract_zero_order import AbstractZeroOrder
from csmt.config import CSMT_NUMPY_DTYPE

np.random.seed(10)
class ZONESMethod(AbstractZeroOrder):

    def gradient_estimation(self,mu,q,x,kappa,target_label,const):
        """
        :param q: number of random direction vectors
        :param mu: key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet
        :param eps_step: 
        :param init_const: regularization parameter prior to attack loss

        - Black-box adversarial attacks with limited queries and information
        | Paper link: http://proceedings.mlr.press/v80/ilyas18a/ilyas18a.pdf
        To estimate the gradient, Ilyas et al.  use Natural Evolutionary Strategies (NES), a method for derivative-free optimization based on the idea of a search distribution $\pi(\theta|x)$.
        Rather than maximizing an objective function $L(x)$ directly, NES maximizes the expected value of the loss function under the search distribution.
        Ilyas et al. employ antithetic sampling to generate a population of $\delta_i$ values: instead of generating $n$ values $\delta_{i} \sim \mathcal{N}(0, I)$, 
        they sample Gaussian noise for $i \in\left\{1, \ldots, \frac{n}{2}\right\}$ and set $\delta_{j}=-\delta_{n-j+1}$ for $j \in\left\{\left(\frac{n}{2}+1\right), \ldots, n\right\}$.
        Evaluating the gradient with a population of $n$ points sampled under this scheme yields the following variancereduced gradient estimate:
        $$
       \nabla \mathbb{E}[L(\theta)] \approx \frac{1}{\sigma n} \sum_{i=1}^{n} \delta_{i} L\left(\theta+\sigma \delta_{i}\right)
        $$
        """
        sigma = 100
        q_prime = int(np.ceil(q/2))
        grad_est=0
        d=x.shape[1]
        logging.debug(d)
        for i in range(q_prime):
            u = np.random.normal(0, sigma, (1,d)).astype(CSMT_NUMPY_DTYPE)
            u_norm = np.linalg.norm(u)
            u = u/u_norm
            # gradient evaluation + -
            f_tmp1, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,x)
            f_tmp2, ignore = self.function_evaluation_cons(x-mu*u,kappa,target_label,const,x)
            grad_est=grad_est+ (d/q)*u*(f_tmp1-f_tmp2)/(2*mu)
            logging.debug(grad_est)
        return grad_est
