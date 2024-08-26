##
"""
This module implements the white-box attack `DeepFool`.

| Paper link: https://arxiv.org/abs/1511.04599
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from tqdm.auto import trange
from typing import Optional, Union, TYPE_CHECKING

from csmt.config import CSMT_NUMPY_DTYPE
from csmt.estimators.estimator import BaseEstimator
from csmt.estimators.classification.classifier import ClassGradientsMixin
from csmt.attacks.attack import EvasionAttack

logger = logging.getLogger(__name__)


class DeepFool(EvasionAttack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2015).

    | Paper link: https://arxiv.org/abs/1511.04599
    """

    attack_params = EvasionAttack.attack_params + [
        "max_iter",
        "epsilon",
        "nb_grads",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        max_iter: int = 100,
        epsilon: float = 1e-6,
        nb_grads: int = 10,
        batch_size: int = 1,
        verbose: bool = False,
        norm: Union[int, float, str] = 1,
    ) -> None:
        """
        Create a DeepFool attack instance.

        :param classifier: A trained classifier.
        :param max_iter: The maximum number of iterations.
        :param epsilon: Overshoot parameter.
        :param nb_grads: The number of class gradients (top nb_grads w.r.t. prediction) to compute. This way only the
                         most likely classes are considered, speeding up the computation.
        :param batch_size: Batch size
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.norm = norm
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.nb_grads = nb_grads
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        if self.estimator.clip_values is None:
            logger.warning(
                "The `clip_values` attribute of the estimator is `None`, therefore this instance of DeepFool will by "
                "default generate adversarial perturbations scaled for input values in the range [0, 1] but not clip "
                "the adversarial example."
            )
    def _apply_norm(self,grad, object_type=False):
        tol = 10e-8
        if self.norm in [np.inf, "inf"]:
            grad = np.sign(grad)
        elif self.norm == 1:
            if not object_type:
                ind = tuple(range(1, len(grad.shape)))
            else:
                ind = None
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            # print((np.sum(np.abs(grad), axis=ind, keepdims=True) + tol))
            # print(np.sum(np.abs(grad[0])))

        elif self.norm == 2:
            if not object_type:
                ind = tuple(range(1, len(grad.shape)))
            else:
                ind = None
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
        return grad

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        # 初始化对抗路径
        x_adv_path=np.zeros((x.shape[0],2,x.shape[1]))
        x_adv_path[:,0]=x

        x_adv = x.astype(CSMT_NUMPY_DTYPE)
        preds = self.estimator.predict(x, batch_size=self.batch_size)

        # if is_probability(preds[0]):
        #     logger.warning(
        #         "It seems that the attacked model is predicting probabilities. DeepFool expects logits as model output "
        #         "to achieve its full attack strength."
        #     )

        # Determine the class labels for which to compute the gradients
        use_grads_subset = self.nb_grads < self.estimator.nb_classes
        if use_grads_subset:
            # TODO compute set of unique labels per batch
            grad_labels = np.argsort(-preds, axis=1)[:, : self.nb_grads]
            labels_set = np.unique(grad_labels)
        else:
            labels_set = np.arange(self.estimator.nb_classes)
        sorter = np.arange(len(labels_set))

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Compute perturbation with implicit batching
        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="DeepFool", disable=not self.verbose
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2].copy()

            # Get predictions and gradients for batch
            f_batch = preds[batch_index_1:batch_index_2]
            fk_hat = np.argmax(f_batch, axis=1)
            if use_grads_subset:
                # Compute gradients only for top predicted classes
                grd = np.array([self.estimator.class_gradient(batch, label=_) for _ in labels_set])
                grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
            else:
                # Compute gradients for all classes
                grd = self.estimator.class_gradient(batch)

            # Get current predictions
            active_indices = np.arange(len(batch))
            current_step = 0
            while active_indices.size > 0 and current_step < self.max_iter:
                # Compute difference in predictions and gradients only for selected top predictions
                labels_indices = sorter[np.searchsorted(labels_set, fk_hat, sorter=sorter)]
                grad_diff = grd - grd[np.arange(len(grd)), labels_indices][:, None]
                f_diff = f_batch[:, labels_set] - f_batch[np.arange(len(f_batch)), labels_indices][:, None]

                # Choose coordinate and compute perturbation
                norm = np.linalg.norm(grad_diff.reshape(len(grad_diff), len(labels_set), -1), axis=2) + tol
                value = np.abs(f_diff) / norm
                value[np.arange(len(value)), labels_indices] = np.inf
                l_var = np.argmin(value, axis=1)
                absolute1 = abs(f_diff[np.arange(len(f_diff)), l_var])
                draddiff = grad_diff[np.arange(len(grad_diff)), l_var].reshape(len(grad_diff), -1)
                pow1 = pow(np.linalg.norm(draddiff, axis=1), 2,) + tol
                r_var = absolute1 / pow1
                r_var = r_var.reshape((-1,) + (1,) * (len(x.shape) - 1))
                r_var = r_var * grad_diff[np.arange(len(grad_diff)), l_var]

                # Add perturbation and clip result
                if self.estimator.clip_values is not None:
                    batch[active_indices] = np.clip(
                        batch[active_indices]
                        + r_var[active_indices] * (self.estimator.clip_values[1] - self.estimator.clip_values[0]),
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1],
                    )
                else:
                    batch[active_indices] += r_var[active_indices]

                # Recompute prediction for new x
                f_batch = self.estimator.predict(batch)
                fk_i_hat = np.argmax(f_batch, axis=1)

                # Recompute gradients for new x
                if use_grads_subset:
                    # Compute gradients only for (originally) top predicted classes
                    grd = np.array([self.estimator.class_gradient(batch, label=_) for _ in labels_set])
                    grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
                else:
                    # Compute gradients for all classes
                    grd = self.estimator.class_gradient(batch)

                # Stop if misclassification has been achieved
                active_indices = np.where(fk_i_hat == fk_hat)[0]

                current_step += 1

            # Apply overshoot parameter
            x_adv1 = x_adv[batch_index_1:batch_index_2]
            x_adv2 = (1 + self.epsilon) * (batch - x_adv[batch_index_1:batch_index_2])
            x_adv[batch_index_1:batch_index_2] = x_adv1 + x_adv2
            if self.estimator.clip_values is not None:
                np.clip(
                    x_adv[batch_index_1:batch_index_2],
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
                    out=x_adv[batch_index_1:batch_index_2],
                )

        # logger.info(
        #     "Success rate of DeepFool attack: %.2f%%",
        #     100 * compute_success(self.estimator, x, y, x_adv, batch_size=self.batch_size),
        # )
        grad=x_adv-x
        grad=self._apply_norm(grad)
        x_adv=x+grad
        x_adv_path[:,1]=x_adv
        return x_adv,x_adv_path

    def _check_params(self) -> None:
        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.nb_grads, (int, np.int)) or self.nb_grads <= 0:
            raise ValueError("The number of class gradients to compute must be a positive integer.")

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
