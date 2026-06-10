"""Unit tests for model utility functions."""

from __future__ import annotations

import pandas as pd
from skopt.space import Categorical, Integer

from src.ml.models.models_utils import (
    AutoregressiveFeaturesTransformer,
    _auto_adjust_n_iter,
    _extract_param_ranges,
)


class TestModelUtilityHelpers:
    """Unit tests for model utility helper functions."""

    def test_auto_adjust_n_iter_caps_finite_categorical_space(self) -> None:
        search_space = {"strategy": Categorical(["mean", "median"])}

        assert _auto_adjust_n_iter(search_space, requested_iter=10) == 2

    def test_extract_param_ranges_handles_integer_dimensions(self) -> None:
        ranges = _extract_param_ranges({"max_depth": Integer(3, 10)})

        assert ranges["min_params"] == {"max_depth": 3}
        assert ranges["max_params"] == {"max_depth": 10}


class TestAutoregressiveFeaturesTransformer:
    """Unit tests for AutoregressiveFeaturesTransformer."""

    def test_fit_transform_adds_lagged_target_features(self) -> None:
        transformer = AutoregressiveFeaturesTransformer(nb_ar=1, nb_mm=0)
        features = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})
        dates = pd.DataFrame({"timestamp": ["t0", "t1", "t2"]})
        target = pd.Series([1.0, 2.0, 3.0])

        transformed, transformed_dates, transformed_target = transformer.fit_transform(
            features,
            dates,
            target,
        )

        assert transformed["target_ar_1"].tolist() == [1.0, 2.0]
        assert transformed_dates["timestamp"].tolist() == ["t1", "t2"]
        assert transformed_target.tolist() == [2.0, 3.0]
