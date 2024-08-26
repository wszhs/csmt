#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Counterfactual explanations for vision tasks.
"""
import warnings
import numpy as np
from ..base import ExplanationBase, DashFigure


class CFExplanation(ExplanationBase):
    """
    The class for image counterfactual explanations. The counterfactual examples of the input
    instances are stored in a list. Each item in the list is a dict with the following format:
    `{"image": the input image, "label": the predicted label of the input image, "cf":
    the counterfactual example, "cf_label": the predicted label of the counterfactual example}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = []

    def __repr__(self):
        return repr(self.explanations)

    def add(self, image, label, cf, cf_label):
        """
        Adds the generated explanation of one image.

        :param image: The input image.
        :param label: The label of the input.
        :param cf: The counterfactual image.
        :param cf_label: The label of the counterfactual image.
        """
        self.explanations.append({"image": image, "label": label, "cf": cf, "cf_label": cf_label})

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index of an explanation result stored in ``CounterfactualExplanation``.
            When ``index`` is None, the function returns a list of all the explanations.
        :return: The explanation for one specific image (a dict)
            or the explanations for all the instances (a list of dicts).
            Each dict has the following format: `{"image": the input image, "label": the predicted
            label of the input image, "cf": the counterfactual example, "cf_label":
            the predicted label of the counterfactual example}`.
        """
        return self.explanations if index is None else self.explanations[index]

    @staticmethod
    def _rescale(im):
        min_val, max_val = np.min(im), np.max(im)
        im = (im - min_val) / (max_val - min_val + 1e-8) * 255
        if im.ndim == 2:
            im = np.tile(np.expand_dims(im, axis=-1), (1, 1, 3))
        return im.astype(np.uint8)

    def plot(self, index=None, class_names=None, **kwargs):
        """
        Returns a matplotlib figure showing the counterfactual explanations.

        :param index: The index of an explanation result stored in ``CounterfactualExplanation``,
            e.g., it will plot the first explanation result when ``index = 0``. When ``index``
            is None, it returns a figure showing the PNs and PPs for the first 5 instances.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A matplotlib figure plotting counterfactuals.
        """
        import matplotlib.pyplot as plt

        explanations = self.get_explanations(index)
        explanations = (
            {index: explanations} if isinstance(explanations, dict) else {i: e for i, e in enumerate(explanations)}
        )
        indices = sorted(explanations.keys())
        if len(indices) > 5:
            warnings.warn(
                f"There are too many instances ({len(indices)} > 5), " f"so only the first 5 instances are plotted."
            )
            indices = indices[:5]
        if len(indices) == 0:
            return

        num_rows = len(indices)
        num_cols = 3
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        for i, index in enumerate(indices):
            e = explanations[index]
            plt.sca(axes[i, 0])
            plt.imshow(self._rescale(e["image"]))
            plt.title(f"{e['label']}" if class_names is None else f"{class_names[e['label']]}")
            plt.axis("off")

            if e["cf_label"] is not None:
                plt.sca(axes[i, 1])
                plt.imshow(self._rescale(e["cf"]))
                plt.title(f"CF: {e['cf_label']}" if class_names is None else f"CF: {class_names[e['cf_label']]}")
                plt.axis("off")

                plt.sca(axes[i, 2])
                plt.imshow(self._rescale(np.abs(e["cf"] - e["image"])))
                plt.title("Difference")
                plt.axis("off")
        return fig

    def _plotly_figure(self, index, class_names=None, **kwargs):
        import plotly.express as px
        from plotly.subplots import make_subplots

        e = self.explanations[index]
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                f"{e['label']}" if class_names is None else f"{class_names[e['label']]}",
                f"CF: {e['cf_label']}" if class_names is None else f"CF: {class_names[e['cf_label']]}",
                "Difference",
            ],
        )
        img_figure = px.imshow(self._rescale(e["image"]))
        fig.add_trace(img_figure.data[0], row=1, col=1)
        if e["cf_label"] is not None:
            img_figure = px.imshow(self._rescale(e["cf"]))
            fig.add_trace(img_figure.data[0], row=1, col=2)
            img_figure = px.imshow(self._rescale(np.abs(e["cf"] - e["image"])))
            fig.add_trace(img_figure.data[0], row=1, col=3)

        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        return fig

    def plotly_plot(self, index=0, class_names=None, **kwargs):
        """
        Returns a plotly dash figure showing the counterfactual explanations for
        one specific instance.

        :param index: The index of an explanation result stored in ``CounterfactualExplanation``
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure plotting counterfactual examples.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(index, class_names=class_names, **kwargs))

    def ipython_plot(self, index=0, class_names=None, **kwargs):
        """
        Plots counterfactual examples in IPython.

        :param index: The index of an explanation result stored in ``CounterfactualExplanation``
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        return plotly.offline.iplot(self._plotly_figure(index, class_names=class_names, **kwargs))
