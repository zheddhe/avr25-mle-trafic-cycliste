import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from skopt.space import Integer, Categorical, Real
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

SEARCH_SPACES_XGB = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(1e-3, 0.3, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 10.0),
    'reg_alpha': Real(1e-4, 10.0, prior='log-uniform'),  # L1
    'reg_lambda': Real(1e-4, 10.0, prior='log-uniform'),  # L2
    'min_child_weight': Integer(1, 20),
}

logger = logging.getLogger(__name__)


def _auto_adjust_n_iter(search_space: dict, requested_iter: int) -> int:
    total = 1
    for dim in search_space.values():
        if isinstance(dim, Categorical):
            total *= len(dim.categories)
        else:
            # pour Real / Integer : espace infini
            return requested_iter  # on ne limite pas
    return min(requested_iter, total)


def _extract_param_ranges(search_space) -> Dict[str, Dict[str, Any]]:
    """
    Extract min and max values from a BayesSearchCV search space.

    Args:
        search_space: skopt Space or dict (e.g., from BayesSearchCV.search_spaces_)

    Returns:
        Dict[str, Dict[str, Any]] with keys 'min_params' and 'max_params'
    """
    min_params = {}
    max_params = {}

    for param_name, dim in search_space.items():
        if isinstance(dim, Real):
            min_params[param_name] = dim.low
            max_params[param_name] = dim.high
        elif isinstance(dim, Integer):
            min_params[param_name] = dim.low
            max_params[param_name] = dim.high
        elif isinstance(dim, Categorical):
            min_params[param_name] = dim.categories[0]
            max_params[param_name] = dim.categories[-1]
        else:
            raise ValueError(f"Unsupported dimension type: {type(dim)}")

    return {
        "min_params": min_params,
        "max_params": max_params,
    }


def train_test_split_time_aware(
    df: pd.DataFrame,
    timestamp_cols: list,
    target_col: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series]:
    """
    Chronological train/test split, preserving datetime columns for visualization.

    Args:
        df: DataFrame containing all data.
        timestamp_cols: list of columns related to time (e.g., ['_utc', '_local']).
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing.

    Returns:
        X_train, X_train_dates, X_test, X_test_dates, y_train, y_test
    """
    df = df.copy()

    # Extract specifically datetime cols for later usage in time series
    features_dates = df[timestamp_cols].copy()

    # Features/target split
    features = df.drop(columns=timestamp_cols + [target_col])
    target = df[target_col]

    # Chronological split
    n_test = int(len(df) * test_size)
    X_train = features[:-n_test]
    X_train_dates = features_dates[:-n_test]
    X_test = features[-n_test:]
    X_test_dates = features_dates[-n_test:]
    y_train = target[:-n_test]
    y_test = target[-n_test:]

    return X_train, X_train_dates, X_test, X_test_dates, y_train, y_test


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


def recursive_forecast_model(
    pipe: Pipeline,
    ar_transformer: AutoregressiveFeaturesTransformer,
    last_window_df: pd.DataFrame,
    horizon: int,
    target_col: str
) -> List[float]:
    """
    Efficient recursive forecast using AR/MM features and exogenous inputs.

    Args:
        pipe (Pipeline): Trained pipeline (prep + regressor).
        ar_transformer: Fitted AutoregressiveFeaturesTransformer.
        last_window_df (pd.DataFrame): Full base with historical train + test X.
        horizon (int): Number of future steps to forecast.
        timestamp_col (str): Name of datetime column.
        target_col (str): Name of target variable.

    Returns:
        List[float]: Forecasted target values (one per horizon step).
    """
    future_preds = []
    current_df = last_window_df.copy()

    # Prepare history buffer (NumPy for speed)
    required_lag = max(
        ar_transformer.nb_ar,
        ar_transformer.nb_mm * ar_transformer.roll_wind
    )
    recent_y = np.array(
        current_df[target_col].dropna().values, dtype=np.float32
    )

    if len(recent_y) < required_lag:
        raise ValueError(
            f"Insufficient history: need at least {required_lag} values in "
            f"`recent_y`, but only {len(recent_y)} provided."
        )

    # Pre-buffer exogenous features (sans target)
    steps_to_forecast = current_df[current_df[target_col].isna()]
    exog_features = steps_to_forecast.drop(columns=[target_col])
    exog_features = exog_features.reset_index(drop=True)
    forecast_rows = []

    for i in range(horizon):
        exog_row = exog_features.iloc[[i]].copy()

        try:
            X_next = ar_transformer.transform_recursive_step(
                exog_row, recent_y.tolist()
            )
        except Exception as e:
            logger.warning(f"[STEP {i}] Failed to create AR features: {e}")
            break

        X_next_prepped = pipe.named_steps["prep"].transform(X_next)
        y_pred = pipe.named_steps["reg"].predict(X_next_prepped)[0]

        future_preds.append(y_pred)
        recent_y = np.append(recent_y, y_pred)

        exog_row[target_col] = y_pred
        forecast_rows.append(exog_row)

    return future_preds


def train_timeseries_model(
    df_compteur: pd.DataFrame,
    target_col: str = "comptage_horaire",
    timestamp_col: str = "date_et_heure_de_comptage",
    drop_columns: Optional[list[str]] = None,
    temp_feats: list[int] = [0, 0, 1],
    test_ratio: float = 0.2,
    forecast: bool = True,
    iter_grid_search: int = 0,
) -> dict:
    """
    Full training logic on a single counter, with optional AR features.

    Returns a dict containing trained model, train/test data, and predictions.
    """
    logger.info(f"Train timeseries with [df_len={len(df_compteur)}"
                f" | drop_columns={drop_columns}]"
                f" | temp_feats={temp_feats}]"
                f" | test_ratio={test_ratio}]"
                f" | forecast={forecast}]")

    df = df_compteur.copy()

    df = df.sort_values(timestamp_col)

    X_train, X_train_dates, X_test, X_test_dates, y_train, y_test = (
        train_test_split_time_aware(
            df,
            timestamp_cols=[timestamp_col],
            target_col=target_col,
            test_size=test_ratio,
        )
    )

    ar_transformer = None
    if temp_feats[:2] != [0, 0]:
        ar_transformer = AutoregressiveFeaturesTransformer(
            nb_ar=temp_feats[0],
            nb_mm=temp_feats[1],
            roll_wind=temp_feats[2],
        )
        X_train, X_train_dates, y_train = ar_transformer.fit_transform(
            X_train, X_train_dates, y_train
        )
        logger.info(f"AR({temp_feats[0]}) et MM({temp_feats[1]}[{temp_feats[2]}h])"
                    " features are applied on train data")

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

    scaler = StandardScaler()

    preprocessing = ColumnTransformer([
        ("num", scaler, numeric_cols),
        ("cat", OneHotEncoder(
             handle_unknown="ignore",
             # drop='first',  # avoid multicolinearity but introduce warnings
             sparse_output=False
         ), categorical_cols)
    ])

    model = XGBRegressor(random_state=1)
    search_spaces = SEARCH_SPACES_XGB
    if iter_grid_search > 0:
        tscv = TimeSeriesSplit(n_splits=5)
        final_model = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces,
            cv=tscv,
            n_iter=_auto_adjust_n_iter(search_spaces, iter_grid_search),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=1
        )
    else:
        final_model = model

    pipe_model = Pipeline([
        ("prep", preprocessing),
        ("reg", final_model)
    ])
    logger.debug(f"Pipeline Model specs used: {pipe_model}")

    pipe_model.fit(X_train, y_train)
    logger.debug("Model training achieved")

    params = None
    if iter_grid_search > 0:
        fitted_model = pipe_model.named_steps['reg']
        best_params = fitted_model.best_params_
        logger.info(f"Bayesian grid search best params [{best_params}]")
        params = {
            "best_params": best_params,
            **_extract_param_ranges(search_spaces)
        }
    y_train_pred = pipe_model.predict(X_train)
    logger.debug("Predictions on Train data achieved")

    if ar_transformer:
        # Assemble full prediction base:
        # - all train rows with y known
        # - all test rows with y unknown (NaN), but features known
        X_full = pd.concat(
            [X_train, X_test],
            ignore_index=True
        )
        y_full = pd.concat(
            [y_train, pd.Series([np.nan] * len(y_test))],
            ignore_index=True
        )
        dates_full = pd.concat(
            [X_train_dates, X_test_dates],
            ignore_index=True
        )
        last_window_df = X_full.copy()
        last_window_df[target_col] = y_full
        last_window_df[timestamp_col] = dates_full
        logger.info(f"recursive predict on an horizon of {len(y_test)} hour(s)")
        y_test_pred = recursive_forecast_model(
            pipe_model,
            ar_transformer,
            last_window_df=last_window_df,
            horizon=len(y_test),
            target_col=target_col
        )
    else:
        y_test_pred = pipe_model.predict(X_test)

    logger.debug("Predictions on Test data achieved")

    return {
        "timestamp_col": timestamp_col,
        "target_col": target_col,
        "ar_transformer": ar_transformer,
        "pipe": pipe_model,
        "params": params,
        "X_train": X_train,
        "X_train_dates": X_train_dates,
        "X_test": X_test,
        "X_test_dates": X_test_dates,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }
