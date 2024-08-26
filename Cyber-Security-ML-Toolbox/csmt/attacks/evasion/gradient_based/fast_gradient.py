

from typing import Optional, Union
import numpy as np
from csmt.config import CSMT_NUMPY_DTYPE
from csmt.attacks.attack import EvasionAttack
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.utils import (
    get_labels_np_array,
    projection,
    random_sphere,
    compute_success,
    check_and_transform_label_format
)

class FastGradientMethod(EvasionAttack):

    def __init__(
        self,
        estimator,
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32
    ) -> None:
        """
        Create a :class:`.FastGradientMethod` instance.
        :param estimator: A trained classifier.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps: Attack step size (input variation).
        :param eps_step: Step size of input variation for minimal perturbation computation.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :param num_random_init: Number of random initialisations within the epsilon ball. For random_init=0 starting at
        the original input. (epsilon 球内的随机初始化次数。 对于 random_init=0 开始于原始输入)
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        super().__init__(estimator=estimator)
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self._targeted = targeted
        self.batch_size = batch_size
        self.num_random_init = num_random_init
        self._project = True
        FastGradientMethod._check_params(self)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values
        :return: An array holding the adversarial examples.
        """

        # Initialize the adversarial path
        x_adv_path=np.zeros((x.shape[0],2,x.shape[1]))
        x_adv_path[:,0]=x

        if isinstance(self.estimator, ClassifierMixin):
            y = check_and_transform_label_format(y, self.estimator.nb_classes)
            if y is None:
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")
                y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore
            y = y / np.sum(y, axis=1, keepdims=True)

            adv_x_best = None
            rate_best = None
            for _ in range(max(1, self.num_random_init)):
                adv_x = self._compute(x, x, y, self.eps, self.eps, self._project, self.num_random_init > 0,**kwargs)

                if self.num_random_init > 1:
                    rate = 100 * compute_success(
                        self.estimator, x, y, adv_x, self.targeted, batch_size=self.batch_size,  # type: ignore
                    )
                    if rate_best is None or rate > rate_best or adv_x_best is None:
                        rate_best = rate
                        adv_x_best = adv_x
                else:
                    adv_x_best = adv_x

        x_adv_path[:,1]=adv_x_best 
        return adv_x_best,x_adv_path

    def _compute(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        project: bool,
        random_init: bool,
        **kwargs
    ) -> np.ndarray:

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(CSMT_NUMPY_DTYPE)
            x_adv = x.astype(CSMT_NUMPY_DTYPE) + random_perturbation
        else:
            x_adv = x.astype(CSMT_NUMPY_DTYPE)

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch_index_2 = min(batch_index_2, x.shape[0])
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]
            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels)

            # Compute batch_eps and batch_eps_step
            batch_eps = eps
            batch_eps_step = eps_step

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, batch_eps_step,**kwargs)

            if project:
                perturbation = projection(x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], batch_eps, self.norm)
                x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv

    def _compute_perturbation(
        self, batch: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        def _apply_norm(grad, object_type=False):
            if self.norm in [np.inf, "inf"]:
                grad = np.sign(grad)
            elif self.norm == 1:
                if not object_type:
                    ind = tuple(range(1, len(batch.shape)))
                else:
                    ind = None
                grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            elif self.norm == 2:
                if not object_type:
                    ind = tuple(range(1, len(batch.shape)))
                else:
                    ind = None
                grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
            return grad

        grad = _apply_norm(grad)

        return grad

    def _apply_perturbation(self, batch: np.ndarray, perturbation: np.ndarray, eps_step: Union[int, float, np.ndarray],**kwargs) -> np.ndarray:
        
        """
        Prevent features whose mask is false from being modified
        This step can also be implemented in the function ‘_apply_feature_mapping’

        """
        c_mask = kwargs.get("c_mask")
        if c_mask is not None:
            perturbation = np.where(c_mask == False, 0.0, perturbation)

        batch = batch + eps_step * perturbation

        batch = self._apply_feature_mapping(batch,**kwargs)

        return batch

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


        # if c_mask.shape[0] != x.shape[1]:  
        #     raise ValueError("Mask shape must be broadcastable to input shape.")
        # if c_range.shape[1] != x.shape[1]:
        #     raise ValueError("Range shape must be broadcastable to input shape.")
        # if c_mask.dtype != bool:
        #         raise ValueError("The `c_mask` has to be bool.")

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

    def _check_params(self) -> None:
        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')

