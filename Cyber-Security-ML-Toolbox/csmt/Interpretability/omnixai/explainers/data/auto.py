#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from typing import Collection, Dict

from ...data.tabular import Tabular
from ..base import AutoExplainerBase


class DataAnalyzer(AutoExplainerBase):
    """
    The class derived from `AutoExplainerBase` for data analysis,
    allowing users to choose multiple explainers and generate
    different explanations at the same time.

    .. code-block:: python

        explainers = TabularExplainer(
            explainers=["imbalance", "mutual"],
            data=data,
            params={"imbalance": {"n_bins": 10}}
        )
        explanations = explainers.explain()
    """

    _MODELS = AutoExplainerBase._EXPLAINERS[__name__.split(".")[2]]

    def __init__(self, explainers: Collection, data: Tabular, params: Dict = None):
        """
        :param explainers: The names or alias of the analyzers to use, e.g.,
            "correlation" for feature correlation analysis, "mutual" for feature importance analysis.
        :param data: The training data used to initialize explainers.
        :param params: A dict containing the additional parameters for initializing each analyzer,
            e.g., `params["imbalance"] = {"param_1": param_1, ...}`.
        """
        super().__init__(
            explainers=explainers,
            mode="data_analysis",
            data=data,
            model=None,
            preprocess=None,
            postprocess=None,
            params=params,
        )

    @staticmethod
    def list_explainers():
        """
        List the supported explainers.
        """
        from tabulate import tabulate
        lists = []
        for _class in DataAnalyzer._MODELS:
            alias = _class.alias if hasattr(_class, "alias") else _class.__name__
            explanation_type = _class.explanation_type \
                if _class.explanation_type != "both" else "global & local"
            lists.append([_class.__module__, _class.__name__, alias, explanation_type])
        table = tabulate(
            lists,
            headers=["Package", "Explainer Class", "Alias", "Explanation Type"],
            tablefmt='orgtbl'
        )
        print(table)
