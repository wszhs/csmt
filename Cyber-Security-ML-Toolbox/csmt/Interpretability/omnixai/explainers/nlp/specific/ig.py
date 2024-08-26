#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The integrated-gradient explainer for NLP tasks.
"""
import numpy as np
from typing import Callable, Dict

from ...base import ExplainerBase
from ....data.text import Text
from ....explanations.text.word_importance import WordImportance
from ....utils.misc import is_torch_available, is_tf_available


def _calculate_integral(inp, baseline, gradients):
    gradients = (gradients[:-1] + gradients[1:]) / 2.0
    avg_grads = np.average(gradients, axis=0)
    integrated_grads = (inp - baseline) * avg_grads
    integrated_grads = np.sum(integrated_grads, axis=-1)
    return integrated_grads


class _IntegratedGradientTorch:
    def __init__(self):
        self.embeddings = None
        self.embedding_layer_inputs = None

    def compute_integrated_gradients(
        self, model, embedding_layer, inputs, output_index, additional_inputs=None, steps=50
    ):
        import torch

        assert inputs.shape[0] == 1, "The batch size of `inputs` should be 1."
        device = next(model.parameters()).device
        hooks = []

        model.eval()
        all_inputs = (inputs,)
        if additional_inputs is not None:
            all_inputs += (additional_inputs,) if not isinstance(additional_inputs, tuple) else additional_inputs

        try:
            # Forward pass for extracting embeddings
            hooks.append(embedding_layer.register_forward_hook(self._embedding_hook))
            model(*all_inputs)
            baselines = np.zeros(self.embeddings.shape)

            # Build the inputs for computing integrated gradient
            alphas = np.linspace(start=0.0, stop=1.0, num=steps, endpoint=True)
            self.embedding_layer_inputs = torch.tensor(
                np.stack([baselines[0] + a * (self.embeddings[0] - baselines[0]) for a in alphas]),
                dtype=torch.get_default_dtype(),
                device=device,
                requires_grad=True,
            )
            all_inputs = self._repeat(all_inputs, num_reps=self.embedding_layer_inputs.shape[0])

            # Compute gradients
            hooks.append(embedding_layer.register_forward_hook(self._embedding_layer_hook))
            predictions = model(*all_inputs)
            if len(predictions.shape) > 1:
                assert output_index is not None, "The model has multiple outputs, the output index cannot be None"
                predictions = predictions[:, output_index]
            gradients = (
                torch.autograd.grad(torch.unbind(predictions), self.embedding_layer_inputs)[0].detach().cpu().numpy()
            )
        finally:
            for hook in hooks:
                hook.remove()
        return _calculate_integral(self.embeddings[0], baselines[0], gradients)

    def _embedding_hook(self, module, inputs, outputs):
        self.embeddings = outputs.detach().cpu().numpy()

    def _embedding_layer_hook(self, module, inputs, outputs):
        return self.embedding_layer_inputs

    @staticmethod
    def _repeat(all_inputs, num_reps):
        return [x.repeat(*((num_reps,) + (1,) * (len(x.shape) - 1))) for x in all_inputs]


class _IntegratedGradientTf:
    def __init__(self):
        self.embeddings = None
        self.embedding_layer_inputs = None

    def compute_integrated_gradients(
        self, model, embedding_layer, inputs, output_index, additional_inputs=None, steps=50
    ):
        import tensorflow as tf

        original_call = embedding_layer.call
        all_inputs = (inputs,)
        if additional_inputs is not None:
            all_inputs += (additional_inputs,) if not isinstance(additional_inputs, tuple) else additional_inputs

        try:
            self._embedding_hook(embedding_layer)
            model(*all_inputs)
            self.embeddings = embedding_layer.res.numpy()
            baselines = np.zeros(self.embeddings.shape)

            # Build the inputs for computing integrated gradient
            alphas = np.linspace(start=0.0, stop=1.0, num=steps, endpoint=True)
            self.embedding_layer_inputs = tf.convert_to_tensor(
                np.stack([baselines[0] + a * (self.embeddings[0] - baselines[0]) for a in alphas]),
                dtype=tf.keras.backend.floatx(),
            )
            all_inputs = [
                tf.tile(x, (self.embedding_layer_inputs.shape[0],) + (1,) * (len(x.shape) - 1)) for x in all_inputs
            ]

            # Compute gradients
            with tf.GradientTape() as tape:
                self._embedding_layer_hook(embedding_layer, tape)
                predictions = model(*all_inputs)
                if len(predictions.shape) > 1:
                    assert output_index is not None, "The model has multiple outputs, the output index cannot be None"
                    predictions = predictions[:, output_index]
                gradients = tape.gradient(predictions, embedding_layer.res).numpy()
        finally:
            self._remove_hook(embedding_layer, original_call)
        return _calculate_integral(self.embeddings[0], baselines[0], gradients)

    def _embedding_hook(self, layer):
        def _hook(func):
            def wrapper(*args, **kwargs):
                layer.res = func(*args, **kwargs)
                return layer.res

            return wrapper

        layer.call = _hook(layer.call)

    def _embedding_layer_hook(self, layer, tape):
        def _hook(func):
            def wrapper(*args, **kwargs):
                layer.res = self.embedding_layer_inputs
                tape.watch(layer.res)
                return layer.res

            return wrapper

        layer.call = _hook(layer.call)

    @staticmethod
    def _remove_hook(layer, original_call):
        layer.call = original_call
        delattr(layer, "res")


class IntegratedGradientText(ExplainerBase):
    """
    The integrated-gradient explainer for NLP tasks.
    If using this explainer, please cite the original work: https://github.com/ankurtaly/Integrated-Gradients.
    """

    explanation_type = "local"
    alias = ["ig", "integrated_gradient"]

    def __init__(
        self,
        model,
        embedding_layer,
        preprocess_function: Callable,
        mode: str = "classification",
        id2token: Dict = None,
        **kwargs,
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param embedding_layer: The embedding layer in the model, which can be
            `tf.keras.layers.Layer` or `torch.nn.Module`.
        :param preprocess_function: The pre-processing function that converts the raw inputs
            into the inputs of ``model``. The first output of ``preprocess_function`` must
            be the token ids.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param id2token: The mapping from token ids to tokens.
        """
        super().__init__()
        assert preprocess_function is not None, (
            "`preprocess_function` cannot be None, which converts a `Text` " "instance into the inputs of `model`."
        )
        self.mode = mode
        self.model = model
        self.embedding_layer = embedding_layer
        self.preprocess_function = preprocess_function
        self.id2token = id2token

        self.ig_class = None
        if is_torch_available():
            import torch.nn as nn

            if isinstance(model, nn.Module):
                self.ig_class = _IntegratedGradientTorch
                self.model_type = "torch"
        if self.ig_class is None and is_tf_available():
            import tensorflow as tf

            if isinstance(model, tf.keras.Model):
                self.ig_class = _IntegratedGradientTf
                self.model_type = "tf"
        if self.ig_class is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")

    def _preprocess(self, X: Text):
        inputs = self.preprocess_function(X)
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        if self.model_type == "torch":
            import torch

            device = next(self.model.parameters()).device
            torch_inputs = []
            for x in inputs:
                if isinstance(x, (np.ndarray, list)):
                    x = torch.tensor(x)
                torch_inputs.append(x.to(device))
            return tuple(torch_inputs)
        else:
            import tensorflow as tf

            tf_inputs = []
            for x in inputs:
                if isinstance(x, (np.ndarray, list)):
                    x = tf.convert_to_tensor(x)
                tf_inputs.append(x)
            return tuple(tf_inputs)

    def explain(self, X: Text, y=None, **kwargs) -> WordImportance:
        """
        Generates the word/token-importance explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when ``y = None``.
        :param kwargs: Additional parameters, e.g., ``steps`` for
            `IntegratedGradient.compute_integrated_gradients`.
        :return: The explanations for all the instances, e.g., word/token importance scores.
        """
        steps = kwargs.get("steps", 50)
        explanations = WordImportance(mode=self.mode)

        inputs = self._preprocess(X)
        if self.mode == "classification":
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(X))]
                else:
                    assert len(X) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of images in X."
                    )
            else:
                scores = (
                    self.model(*inputs).detach().cpu().numpy()
                    if self.model_type == "torch"
                    else self.model(*inputs).numpy()
                )
                y = np.argmax(scores, axis=1).astype(int)

        for i, instance in enumerate(X):
            output_index = y[i] if y is not None else None
            inputs = self._preprocess(instance)
            scores = self.ig_class().compute_integrated_gradients(
                model=self.model,
                embedding_layer=self.embedding_layer,
                inputs=inputs[0],
                output_index=output_index,
                additional_inputs=None if len(inputs) == 1 else inputs[1:],
                steps=steps,
            )
            tokens = inputs[0].detach().cpu().numpy() if self.model_type == "torch" else inputs[0].numpy()
            explanations.add(
                instance=instance.to_str(),
                target_label=y[i] if y is not None else None,
                tokens=tokens[0] if self.id2token is None else [self.id2token[t] for t in tokens[0]],
                importance_scores=scores,
            )
        return explanations
