#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Counterfactual explanations.
"""
import numpy as np
import pandas as pd
from ..base import ExplanationBase, DashFigure


class CFExplanation(ExplanationBase):
    """
    The class for counterfactual explanation results. It uses a list to store
    the generated counterfactual examples. Each item in the list is a dict with
    the following format: `{"query": the original input instance, "counterfactual":
    the generated counterfactual examples}`. Both "query" and "counterfactual" are
    pandas dataframes with an additional column "label" which stores the predicted
    labels of these instances.
    """

    def __init__(self):
        super().__init__()
        self.explanations = []

    def __repr__(self):
        return repr(self.explanations)

    def add(self, query, cfs, **kwargs):
        """
        Adds the generated explanation corresponding to one instance.

        :param query: The instance to explain.
        :param cfs: The generated counterfactual examples.
        :param kwargs: Additional information to store.
        """
        e = {"query": query, "counterfactual": cfs}
        e.update(kwargs)
        self.explanations.append(e)

    def get_explanations(self, index=None):
        """
        Gets the generated counterfactual explanations.

        :param index: The index of an explanation result stored in ``CFExplanation``.
            When it is None, it returns a list of all the explanations.
        :return: The explanation for one specific instance (a dict)
            or all the explanations for all the instances (a list). Each dict has
            the following format: `{"query": the original input instance, "counterfactual":
            the generated counterfactual examples}`. Both "query" and "counterfactual" are
            pandas dataframes with an additional column "label" which stores the predicted
            labels of these instances.
        :rtype: Union[Dict, List]
        """
        return self.explanations if index is None else self.explanations[index]

    def _get_changed_columns(self, query, cfs):
        """
        Gets the differences between the instance and the generated counterfactual examples.

        :param query: The input instance.
        :param cfs: The counterfactual examples.
        :return: The feature columns that have been changed in ``cfs``.
        :rtype: List
        """
        columns = []
        for col in query.columns:
            u = query[[col]].values[0]
            for val in cfs[[col]].values:
                if val != u:
                    columns.append(col)
                    break
        return columns

    @staticmethod
    def _plot(plt, index, df, font_size, bar_width=0.4):
        """
        Plots a table showing the generated counterfactual examples.
        """
        rows = [f"Instance {index}"] + [f"CF {k}" for k in range(1, df.shape[0])]
        counts = np.zeros(len(df.columns))
        for i in range(df.shape[1] - 1):
            for j in range(1, df.shape[0]):
                counts[i] += int(df.values[0, i] != df.values[j, i])

        plt.bar(np.arange(len(df.columns)) + 0.5, counts, bar_width)
        table = plt.table(cellText=df.values, rowLabels=rows, colLabels=df.columns, loc="bottom")
        plt.subplots_adjust(left=0.1, bottom=0.25)
        plt.ylabel("The number of feature changes")
        plt.yticks(np.arange(max(counts)))
        plt.xticks([])
        plt.title(f"Instance {index}: Counterfactual Examples")
        plt.grid()

        # Highlight the differences between the query and the CF examples
        for k in range(df.shape[1]):
            table[(0, k)].set_facecolor("#C5C5C5")
            table[(1, k)].set_facecolor("#E2DED0")
        for j in range(1, df.shape[0]):
            for k in range(df.shape[1] - 1):
                if df.values[0][k] != df.values[j][k]:
                    table[(j + 1, k)].set_facecolor("#56b5fd")

        # Change the font size if `font_size` is set
        if font_size is not None:
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)

    def plot(self, index=None, class_names=None, font_size=10, **kwargs):
        """
        Returns a list of matplotlib figures showing the explanations of
        one or the first 5 instances.

        :param index: The index of an explanation result stored in ``CFExplanation``. For
            example, it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it plots the explanations of the first 5 instances.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param font_size: The font size of table entries.
        :return: A list of matplotlib figures plotting counterfactual examples.
        """
        import warnings
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

        figures = []
        for i, index in enumerate(indices):
            fig = plt.figure()
            figures.append(fig)
            exp = explanations[index]
            if exp["counterfactual"] is None:
                continue
            if len(exp["query"].columns) > 5:
                columns = self._get_changed_columns(exp["query"], exp["counterfactual"])
            else:
                columns = exp["query"].columns
            query, cfs = exp["query"][columns], exp["counterfactual"][columns]
            df = pd.concat([query, cfs], axis=0)
            if class_names is not None:
                df["label"] = [class_names[label] for label in df["label"].values]
            self._plot(plt, index, df, font_size)
        return figures

    def plotly_plot(self, index=0, class_names=None, **kwargs):
        """
        Plots the generated counterfactual examples in Dash.

        :param index: The index of an explanation result stored in ``CFExplanation``,
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure showing the counterfactual examples.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."

        exp = self.explanations[index]
        if exp["counterfactual"] is None:
            return DashFigure(self._plotly_table(exp["query"], None))

        if len(exp["query"].columns) > 5:
            columns = self._get_changed_columns(exp["query"], exp["counterfactual"])
        else:
            columns = exp["query"].columns
        query, cfs = exp["query"][columns], exp["counterfactual"][columns]
        df = pd.concat([query, cfs], axis=0)
        if class_names is not None:
            df["label"] = [class_names[label] for label in df["label"].values]
        return DashFigure(self._plotly_table(df.iloc[0:1], df.iloc[1:]))

    def ipython_plot(self, index=0, class_names=None, **kwargs):
        """
        Plots the generated counterfactual examples in IPython.

        :param index: The index of an explanation result stored in ``CFExplanation``,
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        """
        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        import plotly
        import plotly.figure_factory as ff

        exp = self.explanations[index]
        if exp["counterfactual"] is None:
            return None
        if len(exp["query"].columns) > 5:
            columns = self._get_changed_columns(exp["query"], exp["counterfactual"])
        else:
            columns = exp["query"].columns
        query, cfs = exp["query"][columns], exp["counterfactual"][columns]
        df = pd.concat([query, cfs], axis=0)
        if class_names is not None:
            df["label"] = [class_names[label] for label in df["label"].values]
        plotly.offline.iplot(ff.create_table(df.round(4)))

    @staticmethod
    def _plotly_table(query, cfs):
        """
        Plots a table showing the generated counterfactual examples.
        """
        from dash import dash_table
        feature_columns = query.columns
        columns = [{"name": "#", "id": "#"}] + [{"name": c, "id": c} for c in feature_columns]

        highlights = []
        query = query.values
        if cfs is not None:
            cfs = cfs.values
            for i, cf in enumerate(cfs):
                for j in range(len(cf) - 1):
                    if query[0][j] != cf[j]:
                        highlights.append((i, j))

        data = []
        for x in query:
            data.append({c: d for c, d in zip(feature_columns, x)})
        data.append({c: "-" for c in feature_columns})
        if cfs is not None:
            for x in cfs:
                data.append({c: d for c, d in zip(feature_columns, x)})
        for i, d in enumerate(data):
            if i == 0:
                d.update({"#": "Query"})
            elif i == 1:
                d.update({"#": "-"})
            else:
                d.update({"#": "CF {}".format(i - 1)})

        style_data_conditional = [{"if": {"row_index": 0}, "backgroundColor": "rgb(240, 240, 240)"}]
        for i, j in highlights:
            c = feature_columns[j]
            cond = {
                "if": {"filter_query": "{{{0}}} != ''".format(c), "column_id": c, "row_index": i + 2},
                "backgroundColor": "dodgerblue",
            }
            style_data_conditional.append(cond)

        table = dash_table.DataTable(
            id="table",
            columns=columns,
            data=data,
            style_header_conditional=[{"textAlign": "center"}],
            style_cell_conditional=[{"textAlign": "center"}],
            style_data_conditional=style_data_conditional,
            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
            style_table={"overflowX": "scroll"},
        )
        return table
