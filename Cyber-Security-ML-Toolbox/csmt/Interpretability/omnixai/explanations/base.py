#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from abc import abstractmethod
from ..utils.misc import AutodocABCMeta


class ExplanationBase(metaclass=AutodocABCMeta):
    """
    Abstract base class for explanation results.
    """

    @abstractmethod
    def get_explanations(self, **kwargs):
        """
        Gets the generated explanations.

        :return: A dict or a list containing the explanations.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self, **kwargs):
        """
        Returns a matplotlib figure showing the explanations.

        :return: A matplotlib figure.
        """
        raise NotImplementedError

    @abstractmethod
    def plotly_plot(self, **kwargs):
        """
        Returns a plotly dash component showing the explanations.

        :return: A plotly dash component.
        """
        raise NotImplementedError

    @abstractmethod
    def ipython_plot(self, **kwargs):
        """
        Plots figures in IPython.
        """
        raise NotImplementedError


class DashFigure:
    def __init__(self, component):
        self.component = component
        self.content = self.to_html_div()

    def show(self, **kwargs):
        import dash

        app = dash.Dash()
        app.layout = self.content
        app.run_server(**kwargs)

    def to_html_div(self, id=None):
        import plotly
        from dash import html, dcc, dash_table

        id = str(id)
        if isinstance(self.component, (tuple, list)):
            return html.Div(children=list(self.component), id=id)
        elif isinstance(self.component, html.Div):
            return self.component
        elif isinstance(self.component, dash_table.DataTable):
            return html.Div([self.component], id=id)
        elif isinstance(self.component, plotly.graph_objs.Figure):
            return html.Div([dcc.Graph(figure=self.component, id=id)], id=f"div_{id}")
        else:
            raise ValueError(f"The type of `component` ({type(self.component)}) "
                             f"" f"is not supported by DashFigure.")

    def to_html(self):
        import plotly

        return plotly.io.to_html(self.content)


class PredictedResults(ExplanationBase):
    """
    The class for prediction results.
    """

    def __init__(self, predictions):
        """
        :param predictions: For classfication, ``predictions`` are the predicted class probabilities.
            For regression, ``predictions`` are the predicted values.
        """
        super().__init__()
        if predictions.ndim == 2:
            labels, values = [], []
            sorted_indices = np.argsort(predictions)
            for i in range(predictions.shape[0]):
                top_labels = sorted_indices[i, ::-1][:5]
                labels.append(top_labels)
                values.append([predictions[i, label] for label in top_labels])
            self.results = {"labels": labels, "values": values}
        else:
            self.results = {"labels": None, "values": predictions}

    def get_explanations(self):
        """
        Gets the prediction results.

        :return: The prediction results.
        :rtype: Dict
        """
        return self.results

    def plot(self, index=None, class_names=None, max_num_subplots=4, **kwargs):
        """
        Returns a matplotlib figure showing the predictions.

        :param index: The index of the instance. When it is None,
            it returns a figure with ``max_num_subplots`` subplots where each subplot
            plots the feature importance scores for one instance.
        :param class_names: The class names.
        :param max_num_subplots: The maximum number of subplots in the figure.
        :return: A matplotlib figure plotting the predictions.
        """
        import matplotlib.pyplot as plt

        labels = self.results["labels"]
        values = self.results["values"]
        if index is not None:
            values = values[index: index + 1]
            if labels is not None:
                labels = labels[index: index + 1]
        if max_num_subplots is not None:
            values = values[:max_num_subplots]
            if labels is not None:
                labels = labels[:max_num_subplots]

        num_rows = int(np.round(np.sqrt(len(values))))
        num_cols = int(np.ceil(len(values) / num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        for i in range(len(values)):
            row, col = divmod(i, num_cols)
            plt.sca(axes[row, col])
            if labels is None:
                fnames, scores = ["value"], [values[i]]
                positions = [0.5]
            else:
                fnames, scores = labels[i], values[i]
                if class_names is not None:
                    fnames = [class_names[f] for f in fnames]
                else:
                    fnames = [str(f) for f in fnames]
                positions = np.arange(len(values[i])) + 0.5
            plt.barh(positions[::-1], scores, align="center")
            plt.yticks(positions[::-1], fnames)
            plt.title(f"Instance {i}")
        return fig

    def _plotly_figure(self, index, class_names=None, **kwargs):
        import plotly.express as px

        values = self.results["values"][index]
        labels = self.results["labels"]
        if labels is None:
            fnames, scores = ["value"], [values]
        else:
            fnames, scores = labels[index], values
            fnames = [class_names[f] for f in fnames] if class_names is not None else [str(f) for f in fnames]
        fig = px.bar(
            y=fnames[::-1],
            x=scores[::-1],
            orientation="h",
            labels={"x": "Predicted values", "y": "Labels"},
            title=f"Instance {index}",
            color_discrete_map={True: "#008B8B", False: "#DC143C"},
        )
        return fig

    def plotly_plot(self, index, class_names=None, **kwargs):
        """
        Returns a plotly dash figure showing the predictions for one specific instance.

        :param index: The index of the instance which cannot be None.
        :param class_names: The class names.
        :return: A plotly dash figure plotting the predictions.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(index, class_names=class_names, **kwargs))

    def ipython_plot(self, index, class_names=None, **kwargs):
        """
        Plots prediction results in IPython.

        :param index: The index of the instance which cannot be None.
        :param class_names: The class names.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        return plotly.offline.iplot(self._plotly_figure(index, class_names=class_names, **kwargs))
