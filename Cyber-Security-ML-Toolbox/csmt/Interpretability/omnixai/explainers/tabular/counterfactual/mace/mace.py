#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Model-Agnostic Counterfactual Explanation (MACE) for tabular data.
"""
import numpy as np
import pandas as pd
from typing import List, Callable, Union

from ...base import ExplainerBase
from .....data.tabular import Tabular

from .retrieval import CFRetrieval
from .gld import GLD
from .greedy import Greedy
from .diversify import DiversityModule
from .refine import BinarySearchRefinement
from .....explanations.tabular.counterfactual import CFExplanation


class MACEExplainer(ExplainerBase):
    """
    The Model-Agnostic Counterfactual Explanation (MACE) developed by Yang et al. Please
    cite the paper `MACE: An Efficient Model-Agnostic Framework for Counterfactual Explanation`.
    It supports most black-box models for classification whose input features can either be
    categorical or continuous-valued.
    """

    explanation_type = "local"
    alias = ["mace"]

    def __init__(
        self,
        training_data: Tabular,
        predict_function: Callable,
        mode: str = "classification",
        ignored_features: List = None,
        **kwargs,
    ):
        """
        :param training_data: The data used to initialize a MACE explainer. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, ``training_data`` can be its subset by applying
            `omnixai.sampler.tabular.Sampler.subsample`.
        :param predict_function: The prediction function corresponding to the model to explain.
            The model should be a classifier, the outputs of the ``predict_function``
            are the class probabilities.
        :param mode: The task type can be `classification` only.
        :param ignored_features: The features ignored in generating counterfactual examples.
        :param kwargs: Additional parameters used in `CFRetrieval` and `GLD`. For more information, please
            refer to the classes `mace.retrieval.CFRetrieval` and `mace.gld.GLD`.
        """
        super().__init__()
        assert mode == "classification", "MACE supports classification tasks only."
        self.training_data = training_data.remove_target_column()
        self.predict_function = predict_function
        self.ignored_features = ignored_features

        self.recall = CFRetrieval(training_data, predict_function, ignored_features, **kwargs)
        self.gld = GLD(training_data, predict_function, **kwargs)
        self.greedy = Greedy(training_data, predict_function)
        self.diversity = DiversityModule(training_data, predict_function)
        self.refinement = BinarySearchRefinement(training_data, predict_function)

    def explain(self, X: Tabular, y: Union[List, np.ndarray] = None, max_number_examples: int = 5) -> CFExplanation:
        """
        Generates counterfactual explanations.

        :param X: A batch of input instances.
        :param y: A batch of the desired labels, which should be different from the predicted labels of ``X``.
            If ``y = None``, the desired labels will be the labels different from the predicted labels of ``X``.
        :param max_number_examples: The maximum number of the generated counterfactual
            examples per class for each input instance.
        :return: A CFExplanation object containing the generated explanations.
        """
        if y is not None:
            assert len(y) == X.shape[0], (
                f"The length of `y` should equal the number of instances in `X`, " f"got {len(y)} != {X.shape[0]}"
            )

        X = X.remove_target_column()
        scores = self.predict_function(X)
        labels = np.argmax(scores, axis=1)
        num_classes = scores.shape[1]

        explanations = CFExplanation()
        for i in range(X.shape[0]):
            x = X.iloc(i)
            label = int(labels[i])
            if y is None or y[i] == label:
                desired_labels = [z for z in range(num_classes) if z != label]
            else:
                desired_labels = [int(y[i])]

            all_cfs = []
            for desired_label in desired_labels:
                # Get candidate features
                candidates, indices = self.recall.get_cf_features(x, desired_label)

                # Find counterfactual examples via GLD
                examples = self.gld.get_cf_examples(x, desired_label, candidates)
                if not examples:
                    # If GLD fails, try to apply the greedy method
                    examples = self.greedy.get_cf_examples(x, desired_label, candidates)

                # Generate diverse counterfactual examples
                if examples:
                    cfs = self.diversity.get_diverse_cfs(x, examples["cfs"], desired_label, k=max_number_examples)
                    cfs = self.refinement.refine(x, cfs, desired_label)
                    cfs_df = cfs.to_pd()
                    cfs_df["label"] = desired_label
                    all_cfs.append(cfs_df)

            instance_df = x.to_pd()
            instance_df["label"] = label
            explanations.add(query=instance_df, cfs=pd.concat(all_cfs) if len(all_cfs) > 0 else None)
        return explanations
