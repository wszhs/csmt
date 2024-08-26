##
"""
This module implements the Basic Iterative Method attack `BasicIterativeMethod` as the iterative version of FGM and
FGSM. This is a white-box attack.

| Paper link: https://arxiv.org/abs/1607.02533
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Union, TYPE_CHECKING

import numpy as np

from csmt.attacks.evasion.gradient_based.projected_gradient_descent import ProjectedGradientDescent

# if TYPE_CHECKING:
#     from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class BasicIterativeMethod(ProjectedGradientDescent):
    """
    The Basic Iterative Method is the iterative version of FGM and FGSM.

    | Paper link: https://arxiv.org/abs/1607.02533
    """

    def __init__(
        self,
        estimator,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
    ) -> None:
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param estimator: A trained classifier.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param num_random_init: BIM和PGD的一个区别
        """
        super().__init__(
            estimator=estimator,
            norm=np.inf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init = num_random_init,
            batch_size=batch_size
        )
