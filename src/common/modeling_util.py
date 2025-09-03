import pandas as pd
import numpy as np


class AutoregressiveFeaturesTransformer:
    """
    Adds autoregressive and rolling average features to X.
    Designed to work with (X, X_dates, y) triplets for time series modeling.
    """

    def __init__(
            self,
            nb_ar: int = 1,
            nb_mm: int = 0,
            roll_wind: int = 2):
        self.nb_ar = nb_ar
        self.nb_mm = nb_mm
        self.roll_wind = roll_wind
        self.fitted_ = False

    def fit_transform(self, X: pd.DataFrame, X_dates: pd.DataFrame,
                      y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Applies AR(N) and rolling features to X and aligns X_dates and y.
        Only past data is used: shift(1) avoids data leakage.

        Returns:
            X_transformed, X_dates_transformed, y_transformed
        """
        df = X.copy()
        if self.nb_ar > 0:
            for ar in range(1, self.nb_ar+1):
                df[f"target_ar_{ar}"] = y.shift(ar)
        if self.nb_mm > 0:
            for s in range(1, self.nb_mm+1):
                df[f"target_mm_{self.roll_wind}_{s}"] = y.shift(1).rolling(
                    window=s*self.roll_wind,
                    center=False,
                ).mean()

        # Drop rows with NaN introduced by shift and rolling
        valid_idx = df.dropna().index
        X_transformed = df.loc[valid_idx].reset_index(drop=True)
        y_transformed = y.loc[valid_idx].reset_index(drop=True)
        dates_transformed = X_dates.loc[valid_idx].reset_index(drop=True)

        self.fitted_ = True
        return X_transformed, dates_transformed, y_transformed

    def transform_with_known_y(self, X: pd.DataFrame, X_dates: pd.DataFrame,
                               y: pd.Series) -> tuple[pd.DataFrame,
                                                      pd.DataFrame, pd.Series]:
        """
        Applies same transformation as fit_transform but on test data.
        Requires full y history (at least window+1 rows) to generate features.

        Returns:
            X_transformed, X_dates_transformed, y_transformed
        """
        if not self.fitted_:
            raise RuntimeError("Must call fit_transform before transform_with_known_y.")

        df = X.copy()
        if self.nb_ar > 0:
            for ar in range(1, self.nb_ar+1):
                df[f"target_ar_{ar}"] = y.shift(ar)
        if self.nb_mm > 0:
            for s in range(1, self.nb_mm+1):
                df[f"target_mm_{self.roll_wind}_{s}"] = y.shift(1).rolling(
                    window=s*self.roll_wind,
                    center=False,
                ).mean()

        # Drop initial rows with NaN to avoid leakage
        valid_idx = df.dropna().index
        X_transformed = df.loc[valid_idx].reset_index(drop=True)
        y_transformed = y.loc[valid_idx].reset_index(drop=True)
        dates_transformed = X_dates.loc[valid_idx].reset_index(drop=True)

        return X_transformed, dates_transformed, y_transformed

    def transform_recursive_step(
        self,
        recent_X: pd.DataFrame,
        recent_y: list[float],
    ) -> pd.DataFrame:
        """
        Generate AR/MM features for a single recursive step.

        Args:
            recent_X (pd.DataFrame): last known row of X (without AR/MM).
            recent_y (list[float]): previous target values (length ≥ max required lag).

        Returns:
            pd.DataFrame: DataFrame with AR/MM features added.
        """
        if not self.fitted_:
            raise RuntimeError(
                "Must call fit_transform before transform_recursive_step."
            )

        if len(recent_y) < self.nb_ar:
            raise ValueError(
                f"Insufficient values in recent_y to generate AR({self.nb_ar}) "
                f"features. Received only {len(recent_y)}."
            )

        row = recent_X.iloc[[-1]].copy()

        if self.nb_ar > 0:
            for ar in range(1, self.nb_ar + 1):
                row[f"target_ar_{ar}"] = recent_y[-ar]
        if self.nb_mm > 0:
            y_series = pd.Series(recent_y)
            for s in range(1, self.nb_mm + 1):
                window = s * self.roll_wind
                if len(y_series) >= window:
                    row[f"target_mm_{self.roll_wind}_{s}"] = y_series[-window:].mean()
                else:
                    row[f"target_mm_{self.roll_wind}_{s}"] = np.nan

        return row
