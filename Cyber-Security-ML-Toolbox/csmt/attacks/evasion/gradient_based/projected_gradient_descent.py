
"""
This module implements the Fast Gradient Method attack. This implementation includes the original Fast Gradient Sign
Method attack and extends it to other norms, therefore it is called the Fast Gradient Method.

| Paper link: https://arxiv.org/abs/1412.6572
"""

import logging
from typing import Optional, Union
import numpy as np
from csmt.config import CSMT_NUMPY_DTYPE
from csmt.attacks.evasion.gradient_based.fast_gradient import FastGradientMethod
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.utils import (
    get_labels_np_array,
     compute_success_array,
    check_and_transform_label_format,
)

logger = logging.getLogger(__name__)

class ProjectedGradientDescent(FastGradientMethod):

    def __init__(
        self,
        estimator,
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        batch_size: int = 32,
        num_random_init: int = 1,
        verbose: bool = True
    ) -> None:
        """
        Create a :class:`.FastGradientMethod` instance.
        :param estimator: A trained classifier.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps: Attack step size (input variation).
        :param eps_step: Step size of input variation for minimal perturbation computation.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        super().__init__(estimator=estimator)
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self._targeted = targeted
        self.batch_size = batch_size
        self.max_iter=max_iter
        self._project = True
        self.num_random_init = num_random_init
        self.verbose=verbose
        ProjectedGradientDescent._check_params(self)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: An array holding the adversarial examples.
        """
        # Initialize the adversarial path
        x_adv_path=np.zeros((x.shape[0],2,x.shape[1]))
        x_adv_path[:,0]=x

        if isinstance(self.estimator, ClassifierMixin):
            # Set up targets
            targets = self._set_targets(x, y)
            # Start to compute adversarial examples
            adv_x = x.astype(CSMT_NUMPY_DTYPE)

            from tqdm.auto import trange

            for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
                for rand_init_num in trange(
                    max(1, self.num_random_init), desc="PGD - Random Initializations", disable=not self.verbose
                ):
                    batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                    batch_index_2 = min(batch_index_2, x.shape[0])
                    batch = x[batch_index_1:batch_index_2]
                    batch_labels = targets[batch_index_1:batch_index_2]

                    for i_max_iter in trange(
                        self.max_iter, desc="PGD - Iterations", leave=False, disable=not self.verbose
                    ):
                        batch = self._compute(
                            batch,
                            x[batch_index_1:batch_index_2],
                            batch_labels,
                            self.eps,
                            self.eps_step,
                            self._project,
                            self.num_random_init > 0 and i_max_iter == 0,
                            **kwargs
                        )

                    if rand_init_num == 0:
                        # initial (and possibly only) random restart: we only have this set of
                        # adversarial examples for now
                        adv_x[batch_index_1:batch_index_2] = np.copy(batch)
                    else:
                        # replace adversarial examples if they are successful
                        attack_success = compute_success_array(
                            self.estimator,
                            x[batch_index_1:batch_index_2],
                            targets[batch_index_1:batch_index_2],
                            batch,
                            self.targeted,
                            batch_size=self.batch_size,
                        )
                        adv_x[batch_index_1:batch_index_2][attack_success] = batch[attack_success]


        x_adv_path[:,1]=adv_x 
        return adv_x,x_adv_path

    def _set_targets(self, x: np.ndarray, y: np.ndarray, classifier_mixin: bool = True) -> np.ndarray:
        """
        Check and set up targets.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :return: The targets.
        """
        if classifier_mixin:
            y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            if classifier_mixin:
                targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets

    def _check_params(self) -> None:
        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')
