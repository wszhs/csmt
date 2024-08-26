##
"""
Module containing ART's exceptions.
"""
from typing import List


class EstimatorError(TypeError):
    """
    Basic exception for errors raised by unexpected estimator types.
    """

    def __init__(self, this_class, class_expected_list: List[str], classifier_given) -> None:
        self.this_class = this_class
        self.class_expected_list = class_expected_list
        self.classifier_given = classifier_given

        classes_expected_message = ""
        for idx, class_expected in enumerate(class_expected_list):
            if idx == 0:
                classes_expected_message += "{0}".format(class_expected)
            else:
                classes_expected_message += " and {0}".format(class_expected)

        self.message = (
            "{0} requires an estimator derived from {1}, "
            "the provided classifier is an instance of {2} and is derived from {3}.".format(
                this_class.__name__,
                classes_expected_message,
                type(classifier_given),
                classifier_given.__class__.__bases__,
            )
        )

    def __str__(self) -> str:
        return self.message
