
"""
reference implementation “https://github.com/Trusted-AI/adversarial-robustness-toolbox”
This module implements abstract base and mixin classes for estimators in CSMT.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from csmt.config import CSMT_NUMPY_DTYPE
from csmt.utils import CLIP_VALUES_TYPE

class BaseEstimator(ABC):
    """
    The abstract base class `BaseEstimator` defines the basic requirements of an estimator in CSMT. 
    The BaseEstimator is the highest abstraction of a machine learning model in CSMT.
    抽象基类定义了csmt中的分类器的基本要求，是csmt中机器学习模型的最高抽象
    """

    estimator_params = [
        "model",
        "clip_values"
    ]

    def __init__(
        self,
        model,
        clip_values
    ):

        self._model = model
        self._clip_values = clip_values
        BaseEstimator._check_params(self)

    def set_params(self, **kwargs) -> None:
        """
        Take a dictionary of parameters and apply checks before setting them as attributes.
        获取参数字典，在设置前进行参数检查
        :param kwargs: A dictionary of attributes.
        """
        for key, value in kwargs.items():
            if key in self.estimator_params:
                if hasattr(BaseEstimator, key) and isinstance(getattr(BaseEstimator, key), property):
                    setattr(self, "_" + key, value)
                else:
                    setattr(self, key, value)
            else:
                raise ValueError("Unexpected parameter `{}` found in kwargs.".format(key))
        self._check_params()

    def get_params(self) -> Dict[str, Any]:
        """
        Get all parameters and their values of this estimator.
        获取所有的评估器的参数
        :return: A dictionary of string parameter names to their value.
        """
        params = dict()
        for key in self.estimator_params:
            params[key] = getattr(self, key)
        return params

    def _check_params(self) -> None:
        """
        check all parameters and their values of this estimator.
        检查所有的参数
        :return:
        """
        if self._clip_values is not None:
            if len(self._clip_values) != 2:
                raise ValueError(
                    "`clip_values` should be a tuple of 2 floats or arrays containing the allowed data range."
                )
            if np.array(self._clip_values[0] >= self._clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")

            if isinstance(self._clip_values, np.ndarray):
                self._clip_values = self._clip_values.astype(CSMT_NUMPY_DTYPE)
            else:
                self._clip_values = np.array(self._clip_values, dtype=CSMT_NUMPY_DTYPE)

    # @abstractmethod：抽象方法，含abstractmethod方法的类不能实例化，继承了含abstractmethod方法的子类必须复写所有abstractmethod装饰的方法，未被装饰的可以不重写
    @abstractmethod
    def predict(self, x, **kwargs) -> Any: 
        """
        Perform prediction of the estimator for input `x`.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :return: Predictions by the model.
        :rtype: Format as produced by the `model`
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y, **kwargs) -> None: 
        """
        Fit the estimator using the training data `(x, y)`.

        :param x: Training data.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        """
        raise NotImplementedError

    # @property 装饰器来创建只读属性
    @property
    def model(self):
        """
        Return the model.

        :return: The model.
        """
        return self._model

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        raise NotImplementedError

    @property
    def clip_values(self) -> Optional["CLIP_VALUES_TYPE"]:
        """
        Return the clip values of the input samples.

        :return: Clip values (min, max).
        """
        return self._clip_values

class LossGradientsMixin(ABC):
    """
    Mixin abstract base class defining additional functionality for estimators providing loss gradients. An estimator
    of this type can be combined with white-box attacks. This mixin abstract base class has to be mixed in with class `BaseEstimator`.
    Mixin 抽象基类为提供损失梯度的估计器定义了附加功能
    """

    @abstractmethod
    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :return: Loss gradients w.r.t. `x` in the same format as `x`.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

class NeuralNetworkMixin(ABC):
    """
    Mixin abstract base class defining additional functionality required for neural network estimators. 
    This base class has to be mixed in with class `BaseEstimator`.
    Mixin 抽象基类定义了神经网络估计器所需的附加功能
    """
    def __init__(self, **kwargs) -> None:
        """
        Initialize a neural network attributes.

        :param channel_index: Index of the axis in samples `x` representing the color channels.
        """
        super().__init__(**kwargs)  # type: ignore

    @abstractmethod
    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Batch size.
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :param nb_epochs: Number of training epochs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of a specific layer for samples `x` where `layer` is the index of the layer between 0 and
        `nb_layers - 1 or the name of the layer. The number of layers can be determined by counting the results
        returned by calling `layer_names`.

        :param x: Samples
        :param layer: Index or name of the layer.
        :param batch_size: Batch size.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        raise NotImplementedError

    @abstractmethod
    def set_learning_phase(self, train: bool) -> None:
        """
        Set the learning phase for the backend framework.

        :param train: `True` if the learning phase is training, otherwise `False`.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    @property
    def learning_phase(self) -> Optional[bool]:
        """
        The learning phase set by the user. Possible values are `True` for training or `False` for prediction and
        `None` if it has not been set by the library. In the latter case, the library does not do any explicit learning
        phase manipulation and the current value of the backend framework is used. If a value has been set by the user
        for this property, it will impact all following computations for model fitting, prediction and gradients.

        :return: Learning phase.
        """
        return self._learning_phase  # type: ignore

    @property
    def layer_names(self) -> Optional[List[str]]:
        """
        Return the names of the hidden layers in the model, if applicable.

        :return: The names of the hidden layers in the model, input and output layers are ignored.

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        return self._layer_names  # type: ignore

class DecisionTreeMixin(ABC):
    """
    Mixin abstract base class defining additional functionality for decision-tree-based estimators. This mixin abstract
    base class has to be mixed in with class `BaseEstimator`.
    """

    @abstractmethod
    def get_trees(self) -> List["Tree"]:
        """
        Get the decision trees.

        :return: A list of decision trees.
        """
        raise NotImplementedError
