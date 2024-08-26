
"""
Module providing convenience functions.
"""
from functools import wraps
import logging
import os
import shutil
import tarfile
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import warnings
import zipfile

import numpy as np
from scipy.special import gammainc
from tqdm.auto import tqdm


if TYPE_CHECKING:
    import torch
    
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES

DATASET_TYPE = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float, float]
CLIP_VALUES_TYPE = Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]


def get_logger():

    logger = logging.getLogger('test_csmt')
    logger.setLevel(logging.DEBUG)
    KZT = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')
    KZT.setFormatter(formatter)
    logger.addHandler(KZT)

    return logger

def get_number_labels(y):
    """
    get the number of labels
    """
    # if isinstance(y[0],np.ndarray):
    #     len_y=len(y[0])
    if len(np.unique(y))<=2:
        len_y=2
    else:
        len_y=len(np.unique(y))
    return len_y
    
def deprecated(end_version: str, *, reason: str = "", replaced_by: str = "") -> Callable:
    """
    Deprecate a function or method and raise a `DeprecationWarning`.
    弃用函数或方法并引发 `DeprecationWarning'

    The `@deprecated` decorator is used to deprecate functions and methods. Several cases are supported. For example
    one can use it to deprecate a function that has become redundant or rename a function. The following code examples
    provide different use cases of how to use decorator.

    .. code-block:: python

      @deprecated("0.1.5", replaced_by="sum")
      def simple_addition(a, b):
          return a + b

    :param end_version: Release version of removal.
    :param reason: Additional deprecation reason.
    :param replaced_by: Function that replaces deprecated function.
    """

    def decorator(function):
        reason_msg = "\n" + reason if reason else reason
        replaced_msg = f" It will be replaced by '{replaced_by}'." if replaced_by else replaced_by
        deprecated_msg = (
            f"Function '{function.__name__}' is deprecated and will be removed in future release {end_version}."
        )

        @wraps(function)
        def wrapper(*args, **kwargs):
            warnings.simplefilter("always", category=DeprecationWarning)
            warnings.warn(
                deprecated_msg + replaced_msg + reason_msg, category=DeprecationWarning, stacklevel=2,
            )
            warnings.simplefilter("default", category=DeprecationWarning)
            return function(*args, **kwargs)

        return wrapper

    return decorator

# ----------------------------------------------------------------------------------------------------- MATH OPERATIONS


def projection(values: np.ndarray, eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]) -> np.ndarray:
    """
    Project `values` on the L_p norm ball of size `eps`.
    :param values: Array of perturbations to clip.
    :param eps: Maximum norm allowed.
    :param norm_p: L_p norm to use for clipping. Only 1, 2, `np.Inf` and "inf" supported for now.
    :return: Values of `values` after projection.
    """
    # Pick a small scalar to avoid division by 0
    tol = 10e-8
    values_tmp = values.reshape((values.shape[0], -1))

    if norm_p == 2:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError("The parameter `eps` of type `np.ndarray` is not supported to use with norm 2.")

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1
        )

    elif norm_p == 1:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError("The parameter `eps` of type `np.ndarray` is not supported to use with norm 1.")

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1,
        )

    elif norm_p in [np.inf, "inf"]:
        if isinstance(eps, np.ndarray):
            eps = eps * np.ones_like(values)
            eps = eps.reshape([eps.shape[0], -1])

        values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)

    else:
        raise NotImplementedError(
            'Values of `norm_p` different from 1, 2, `np.inf` and "inf" are currently not ' "supported."
        )

    values = values_tmp.reshape(values.shape)

    return values


def random_sphere(
    nb_points: int, nb_dims: int, radius: Union[int, float, np.ndarray], norm: Union[int, float, str],
) -> np.ndarray:
    """
    Generate randomly `m x n`-dimension points with radius `radius` and centered around 0.
    随机生成半径为“radius”且以 0 为中心的“m x n”维的点
    :param nb_points: Number of random data points.
    :param nb_dims: Dimensionality of the sphere.
    :param radius: Radius of the sphere.
    :param norm: Current support: 1, 2, np.inf, "inf".
    :return: The generated random sphere.
    """
    if norm == 1:
        if isinstance(radius, np.ndarray):
            raise NotImplementedError(
                "The parameter `radius` of type `np.ndarray` is not supported to use with norm 1."
            )

        a_tmp = np.zeros(shape=(nb_points, nb_dims + 1))
        a_tmp[:, -1] = np.sqrt(np.random.uniform(0, radius ** 2, nb_points))

        for i in range(nb_points):
            a_tmp[i, 1:-1] = np.sort(np.random.uniform(0, a_tmp[i, -1], nb_dims - 1))

        res = (a_tmp[:, 1:] - a_tmp[:, :-1]) * np.random.choice([-1, 1], (nb_points, nb_dims))

    elif norm == 2:
        if isinstance(radius, np.ndarray):
            raise NotImplementedError(
                "The parameter `radius` of type `np.ndarray` is not supported to use with norm 2."
            )

        a_tmp = np.random.randn(nb_points, nb_dims)
        s_2 = np.sum(a_tmp ** 2, axis=1)
        base = gammainc(nb_dims / 2.0, s_2 / 2.0) ** (1 / nb_dims) * radius / np.sqrt(s_2)
        res = a_tmp * (np.tile(base, (nb_dims, 1))).T

    elif norm in [np.inf, "inf"]:
        if isinstance(radius, np.ndarray):
            radius = radius * np.ones(shape=(nb_points, nb_dims))

        res = np.random.uniform(-radius, radius, (nb_points, nb_dims))

    else:
        raise NotImplementedError("Norm {} not supported".format(norm))

    return res


def original_to_tanh(
    x_original: np.ndarray,
    clip_min: Union[float, np.ndarray],
    clip_max: Union[float, np.ndarray],
    tanh_smoother: float = 0.999999,
) -> np.ndarray:
    """
    Transform input from original to tanh space.
    将输入从原始空间转换为 tanh 空间
    :param x_original: An array with the input to be transformed.
    :param clip_min: Minimum clipping value.
    :param clip_max: Maximum clipping value.
    :param tanh_smoother: Scalar for multiplying arguments of arctanh to avoid division by zero.
    :return: An array holding the transformed input.
    """
    x_tanh = np.clip(x_original, clip_min, clip_max)
    x_tanh = (x_tanh - clip_min) / (clip_max - clip_min)
    x_tanh = np.arctanh(((x_tanh * 2) - 1) * tanh_smoother)
    return x_tanh


def tanh_to_original(
    x_tanh: np.ndarray, clip_min: Union[float, np.ndarray], clip_max: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Transform input from tanh to original space.
    将输入从 tanh 转换为原始空间
    :param x_tanh: An array with the input to be transformed.
    :param clip_min: Minimum clipping value.
    :param clip_max: Maximum clipping value.
    :return: An array holding the transformed input.
    """
    return (np.tanh(x_tanh) + 1.0) / 2.0 * (clip_max - clip_min) + clip_min

# --------------------------------------------------------------------------------------------------- LABELS OPERATIONS

def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.
    将标签数组转换为二进制类矩阵
    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=np.int32)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int] = None, return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary
    检查标签格式并在必要时转换为单热编码标签
    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    if labels is not None:
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            if not return_one_hot:
                labels = np.argmax(labels, axis=1)
        elif len(labels.shape) == 2 and labels.shape[1] == 1:
            labels = np.squeeze(labels)
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        elif len(labels.shape) == 1:
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

    return labels


def random_targets(labels: np.ndarray, nb_classes: int) -> np.ndarray:
    """
    Given a set of correct labels, randomly changes some correct labels to target labels different from the original
    ones. These can be one-hot encoded or integers.
    给定一组正确的标签，随机将一些正确的标签更改为与原始标签不同的目标标签
    :param labels: The correct labels.
    :param nb_classes: The number of classes for this model.
    :return: An array holding the randomly-selected target classes, one-hot encoded.
    """
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)

    result = np.zeros(labels.shape)

    for class_ind in range(nb_classes):
        other_classes = list(range(nb_classes))
        other_classes.remove(class_ind)
        in_cl = labels == class_ind
        result[in_cl] = np.random.choice(other_classes)

    return to_categorical(result, nb_classes)


def get_labels_np_array(preds: np.ndarray) -> np.ndarray:
    """
    Returns the label of the most probable class given a array of class confidences.

    :param preds: Array of class confidences, nb of instances as first dimension.
    :return: Labels.
    """
    preds_max = np.amax(preds, axis=1, keepdims=True)
    y = preds == preds_max
    y = y.astype(np.uint8)
    return y


def compute_success_array(
    classifier,
    x_clean: np.ndarray,
    labels: np.ndarray,
    x_adv: np.ndarray,
    targeted: bool = False,
    batch_size: int = 1,
) -> float:
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :param x_clean: Original clean samples.
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :param x_adv: Adversarial samples to be evaluated.
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.
    :param batch_size: Batch size.
    :return: Percentage of successful adversarial samples.
    """
    adv_preds = np.argmax(classifier.predict(x_adv, batch_size=batch_size), axis=1)
    if targeted:
        attack_success = adv_preds == np.argmax(labels, axis=1)
    else:
        preds = np.argmax(classifier.predict(x_clean, batch_size=batch_size), axis=1)
        attack_success = adv_preds != preds

    return attack_success


def compute_success(
    classifier: "CLASSIFIER_TYPE",
    x_clean: np.ndarray,
    labels: np.ndarray,
    x_adv: np.ndarray,
    targeted: bool = False,
    batch_size: int = 1,
) -> float:
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :param x_clean: Original clean samples.
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :param x_adv: Adversarial samples to be evaluated.
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.
    :param batch_size: Batch size.
    :return: Percentage of successful adversarial samples.
    """
    attack_success = compute_success_array(classifier, x_clean, labels, x_adv, targeted, batch_size)
    return np.sum(attack_success) / x_adv.shape[0]


# -------------------------------------------------------------------------------------------------- DATASET OPERATIONS

def load_mnist(raw: bool = False,) -> DATASET_TYPE:
    """
    Loads MNIST dataset from `config.CSMT_DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :return: `(x_train, y_train), (x_test, y_test), min, max`.
    """
    # path = get_file("mnist.npz", path=config.CSMT_DATA_PATH, url="https://s3.amazonaws.com/img-datasets/mnist.npz",)
    path='csmt/datasets/data/mnist/mnist.npz'

    dict_mnist = np.load(path)
    x_train = dict_mnist["x_train"]
    y_train = dict_mnist["y_train"]
    x_test = dict_mnist["x_test"]
    y_test = dict_mnist["y_test"]
    dict_mnist.close()

    # Add channel axis
    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_

def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    nb_classes: int = 10,
    clip_values: Optional["CLIP_VALUES_TYPE"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :param y: Labels.
    :param nb_classes: Number of classes in dataset.
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :return: Rescaled values of `x`, `y`.
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return normalized_x, categorical_y


def _extract(full_path: str, path: str) -> bool:
    archive: Union[zipfile.ZipFile, tarfile.TarFile]
    if full_path.endswith("tar"):
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:")
    elif full_path.endswith("tar.gz"):
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:gz")
    elif full_path.endswith("zip"):
        if zipfile.is_zipfile(full_path):
            archive = zipfile.ZipFile(full_path)
        else:
            return False
    else:
        return False

    try:
        archive.extractall(path)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        raise
    return True


def get_file(filename: str, url: str, path: Optional[str] = None, extract: bool = False, verbose: bool = False) -> str:
    """
    Downloads a file from a URL if it not already in the cache. The file at indicated by `url` is downloaded to the
    path `path` (default is ~/.csmt/data). and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip formats
    can also be extracted. This is a simplified version of the function with the same name in Keras.

    :param filename: Name of the file.
    :param url: Download URL.
    :param path: Folder to store the download. If not specified, `~/.csmt/data` is used instead.
    :param extract: If true, tries to extract the archive.
    :param verbose: If true, print download progress bar.
    :return: Path to the downloaded file.
    """
    if path is None:
        from art import config

        path_ = os.path.expanduser(config.CSMT_DATA_PATH)
    else:
        path_ = os.path.expanduser(path)
    if not os.access(path_, os.W_OK):
        path_ = os.path.join("/tmp", ".art")

    if not os.path.exists(path_):
        os.makedirs(path_)

    if extract:
        extract_path = os.path.join(path_, filename)
        full_path = extract_path + ".tar.gz"
    else:
        full_path = os.path.join(path_, filename)

    # Determine if dataset needs downloading
    download = not os.path.exists(full_path)

    if download:
        logger.info("Downloading data from %s", url)
        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                from six.moves.urllib.error import HTTPError, URLError
                from six.moves.urllib.request import urlretrieve

                # The following two lines should prevent occasionally occurring
                # [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)
                import ssl

                ssl._create_default_https_context = ssl._create_unverified_context

                if verbose:
                    with tqdm() as t:
                        last_block = [0]

                        def progress_bar(blocks: int = 1, block_size: int = 1, total_size: Optional[int] = None):
                            """
                            :param blocks: Number of blocks transferred so far [default: 1].
                            :param block_size: Size of each block (in tqdm units) [default: 1].
                            :param total_size: Total size (in tqdm units). If [default: None] or -1, remains unchanged.
                            """
                            if total_size not in (None, -1):
                                t.total = total_size
                            displayed = t.update((blocks - last_block[0]) * block_size)
                            last_block[0] = blocks
                            return displayed

                        urlretrieve(url, full_path, reporthook=progress_bar)
                else:
                    urlretrieve(url, full_path)

            except HTTPError as exception:
                raise Exception(error_msg.format(url, exception.code, exception.msg)) from HTTPError  # type: ignore
            except URLError as exception:
                raise Exception(error_msg.format(url, exception.errno, exception.reason)) from HTTPError
        except (Exception, KeyboardInterrupt):
            if os.path.exists(full_path):
                os.remove(full_path)
            raise

    if extract:
        if not os.path.exists(extract_path):
            _extract(full_path, path_)
        return extract_path

    return full_path

def performance_diff(
    model1: "CLASSIFIER_TYPE",
    model2: "CLASSIFIER_TYPE",
    test_data: np.ndarray,
    test_labels: np.ndarray,
    perf_function: Union[str, Callable] = "accuracy",
    **kwargs,
) -> float:
    """
    Calculates the difference in performance between two models on the test_data with a performance function.
    使用性能函数计算 test_data 上两个模型之间的性能差异。

    Note: For multi-label classification, f1 scores will use 'micro' averaging unless otherwise specified.

    :param model1: A trained ART classifier.
    :param model2: Another trained ART classifier.
    :param test_data: The data to test both model's performance.
    :param test_labels: The labels to the testing data.
    :param perf_function: The performance metric to be used. One of ['accuracy', 'f1'] or a callable function
           `(true_labels, model_labels[, kwargs]) -> float`.
    :param kwargs: Arguments to add to performance function.
    :return: The difference in performance performance(model1) - performance(model2).
    :raises `ValueError`: If an unsupported performance function is requested.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    model1_labels = model1.predict(test_data)
    model2_labels = model2.predict(test_data)

    if perf_function == "accuracy":
        model1_acc = accuracy_score(test_labels, model1_labels, **kwargs)
        model2_acc = accuracy_score(test_labels, model2_labels, **kwargs)
        return model1_acc - model2_acc

    if perf_function == "f1":
        n_classes = test_labels.shape[1]
        if n_classes > 2 and "average" not in kwargs:
            kwargs["average"] = "micro"
        model1_f1 = f1_score(test_labels, model1_labels, **kwargs)
        model2_f1 = f1_score(test_labels, model2_labels, **kwargs)
        return model1_f1 - model2_f1

    if callable(perf_function):
        return perf_function(test_labels, model1_labels, **kwargs) - perf_function(test_labels, model2_labels, **kwargs)
    raise ValueError("Performance function '{}' not supported".format(str(perf_function)))


# -------------------------------------------------------------------------------------------------------- CUDA SUPPORT


def to_cuda(x: "torch.Tensor") -> "torch.Tensor":
    """
    Move the tensor from the CPU to the GPU if a GPU is available.

    :param x: CPU Tensor to move to GPU if available.
    :return: The CPU Tensor moved to a GPU Tensor.
    """
    from torch.cuda import is_available

    use_cuda = is_available()
    if use_cuda:
        x = x.cuda()
    return x


def from_cuda(x: "torch.Tensor") -> "torch.Tensor":
    """
    Move the tensor from the GPU to the CPU if a GPU is available.

    :param x: GPU Tensor to move to CPU if available.
    :return: The GPU Tensor moved to a CPU Tensor.
    """
    from torch.cuda import is_available

    use_cuda = is_available()
    if use_cuda:
        x = x.cpu()
    return x
