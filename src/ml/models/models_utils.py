import pandas as pd
import numpy as np
import os
import logging
import joblib
import json
from typing import List, Tuple, Dict, Any
from skopt.space import Integer, Categorical, Real
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

SEARCH_SPACES_XGB = {
    'n_estimators': Integer(300, 800),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'subsample': Real(0.7, 1.0),
    'colsample_bytree': Real(0.7, 1.0),
    'gamma': Real(0, 5.0),
    'reg_alpha': Real(1e-4, 1.0, prior='log-uniform'),  # L1
    'reg_lambda': Real(0.1, 5, prior='log-uniform'),  # L2
    'min_child_weight': Integer(1, 10),
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
    df_counter: pd.DataFrame,
    target_col: str = "comptage_horaire",
    timestamp_cols: List[str] = ["date_et_heure_de_comptage_local"],
    temp_feats: list[int] = [0, 0, 1],
    test_ratio: float = 0.2,
    iter_grid_search: int = 0,
) -> dict:
    """
    This function search for the best model using a bayesian gridsearch
    It prepares and split the data between train and test applying the AR/MA features on train
    it predicts on test by using a strict iterative forecast by propagating the predictions
    to apply the AR/MA features on the test part (starting from a window including the train data
    and calculating on a horizon on the full test data provided

    Arguments:
        df_counter: pd.DataFrame, full train and test data to train and then predict upon
        target_col: str, target column of the prediction
        timestamp_cols: List[str], list of timestamp colons to extract from the features
        temp_feats: list[int], temporal autoregressive features and mobile averages to construct
            - 1) nb of AR (exemple if 7 : AR-1/AR-2/.../AR-7 are constructed)
            - 2) nb of MA (exemple if 1 : MA-1 is constructed)
            - 3) size of MA window (exemple : if 24 : MA-X are constructed with multiple of 24 lags)
        test_ratio: float,
        iter_grid_search: int, number of iteration to use to find the best model by gridsearch

    Returns:
        Dict with trained pipeline, train/test refactored data including AR and predictions.
    """
    logger.info(
        f"Train and predict timeseries with [df_len={len(df_counter)}"
        f" | temp_feats={temp_feats}"
        f" | test_ratio={test_ratio}"
        f" | iter_grid_search={iter_grid_search}]"
    )

    # initial copy
    df = df_counter.copy()

    # split between train and test keeping the dates as well
    X_train, X_train_dates, X_test, X_test_dates, y_train, y_test = (
        train_test_split_time_aware(
            df,
            timestamp_cols=timestamp_cols,
            target_col=target_col,
            test_size=test_ratio,
        )
    )

    # Apply autoregressive features (AR/MA) on train data
    ar_transformer = AutoregressiveFeaturesTransformer(
        nb_ar=temp_feats[0],  # number of AR
        nb_mm=temp_feats[1],  # number of MA
        roll_wind=temp_feats[2],  # lag per MA
    )
    X_train, X_train_dates, y_train = ar_transformer.fit_transform(
        X_train, X_train_dates, y_train
    )
    logger.info(
        f"AR({temp_feats[0]}) et MM({temp_feats[1]}[{temp_feats[2]}h])"
        " features are applied on train data"
    )

    # Set up the preprocessing transformers
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    logger.info(f"List of numerical columns : {numeric_cols}")
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    logger.info(f"List of categorical columns : {categorical_cols}")
    scaler = StandardScaler()
    preprocessing = ColumnTransformer([
        ("num", scaler, numeric_cols),
        ("cat", OneHotEncoder(
             handle_unknown="ignore",
             # drop='first',  # avoid multicolinearity but introduce warnings
             sparse_output=False
         ), categorical_cols)
    ])

    # Set up the model and initiale gridsearch if iteration are foreseen
    model = XGBRegressor(random_state=1, n_jobs=-1)
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

    # Define the full preprocessing and model pipeline
    pipe_model = Pipeline([
        ("prep", preprocessing),
        ("reg", final_model)
    ])
    logger.debug(f"Pipeline Model specs used: {pipe_model}")

    # Train the model with that pipeline
    pipe_model.fit(X_train, y_train)
    logger.info("Model training achieved")

    # Collect the best model parameters
    fitted_model = pipe_model.named_steps['reg']
    if iter_grid_search > 0:
        best_params = fitted_model.best_params_
        logger.info(f"Bayesian grid search best params [{best_params}]")
        best_estimator = fitted_model.best_estimator_
        fitted_model_params = best_estimator.get_params()
        logger.info(f"fitted model params [{fitted_model_params}]")
    else:
        best_params = "Not Applicable (no grid search)"
        fitted_model_params = fitted_model.get_params()
        logger.info(f"fitted model params [{fitted_model_params}]")
    params = {
        "best_params": best_params,
        **_extract_param_ranges(search_spaces),
        **fitted_model_params,
    }
    y_train_pred = pipe_model.predict(X_train)
    logger.info("Predictions on train data achieved")

    # Assemble full prediction window and trigger the prediction in forecast mode
    # - last X_train features and y_train target known
    # - all X_test features and not using y_test (left with NaN)
    X_full = pd.concat(
        [X_train, X_test],
        ignore_index=True
    )
    y_full = pd.concat(
        [y_train, pd.Series([np.nan] * len(y_test))],
        ignore_index=True
    )
    last_window_df = X_full.copy()
    last_window_df[target_col] = y_full
    logger.info(f"Recursive predict on an horizon of {len(y_test)} hour(s)")
    y_test_pred = recursive_forecast_model(
        pipe_model,
        ar_transformer,
        last_window_df=last_window_df,
        horizon=len(y_test),
        target_col=target_col
    )
    logger.info("Predictions on test data achieved")

    # Assemble a full prediction dataset based on y_true\y_pred\dates (from train and test)
    y_true = pd.Series(y_train).reset_index(drop=True)
    y_pred = pd.Series(y_train_pred).reset_index(drop=True)
    train_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "forecast_mode": False,
    })
    train_df = pd.concat([X_train_dates.reset_index(drop=True), train_df], axis=1)
    y_true = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(y_test_pred).reset_index(drop=True)
    test_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "forecast_mode": True,
    })
    test_df = pd.concat([X_test_dates.reset_index(drop=True), test_df], axis=1)
    y_full = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Compute train metrics
    metrics = {
        "train": compute_metrics(pd.Series(y_train_pred), y_train),
        "test": compute_metrics(pd.Series(y_test_pred), y_test),
    }

    # provide all generated item in a dictionnary
    return {
        "ar_transformer": ar_transformer,
        "pipe_model": pipe_model,
        "params": params,
        "X_train": X_train,
        "X_train_dates": X_train_dates,
        "X_test": X_test,
        "X_test_dates": X_test_dates,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_full": y_full,
        "metrics": metrics,
    }


def save_artefacts(report: Dict, save_dir):
    # save final training/test/predictions data from the results :
    save_data_path = os.path.join("data", "final", save_dir)
    os.makedirs(save_data_path, exist_ok=True)
    X_train_path = os.path.join(save_data_path, "X_train.csv")
    X_test_path = os.path.join(save_data_path, "X_test.csv")
    X_train_dates_path = os.path.join(save_data_path, "X_train_dates.csv")
    X_test_dates_path = os.path.join(save_data_path, "X_test_dates.csv")
    y_train_path = os.path.join(save_data_path, "y_train.csv")
    y_test_path = os.path.join(save_data_path, "y_test.csv")
    y_train_pred_path = os.path.join(save_data_path, "y_train_pred.csv")
    y_test_pred_path = os.path.join(save_data_path, "y_test_pred.csv")
    y_full_path = os.path.join(save_data_path, "y_full.csv")
    logging.info(
        "Final refined data CSV files saved in {save_data_path}:\n"
        f"{X_train_path}\n"
        f"{X_test_path}\n"
        f"{X_train_dates_path}\n"
        f"{X_test_dates_path}\n"
        f"{y_train_path}\n"
        f"{y_test_path}\n"
        f"{y_train_pred_path}\n"
        f"{y_test_pred_path}"
    )
    report["X_test_dates"].to_csv(X_test_dates_path, index=True)
    report["X_train"].to_csv(X_train_path, index=True)
    report["X_test"].to_csv(X_test_path, index=True)
    report["X_train_dates"].to_csv(X_train_dates_path, index=True)
    report["y_train"].to_csv(y_train_path, index=True)
    report["y_test"].to_csv(y_test_path, index=True)
    pd.DataFrame(
        report["y_train_pred"],
        columns=["comptage_horaire_predit"],
    ).to_csv(y_train_pred_path, index=True)
    pd.DataFrame(
        report["y_test_pred"],
        columns=["comptage_horaire_predit"],
    ).to_csv(y_test_pred_path, index=True)
    report["y_full"].to_csv(y_full_path, index=True)

    # save pipeline model/params/transformer from the result
    save_model_path = os.path.join("models", save_dir)
    os.makedirs(save_model_path, exist_ok=True)
    pipe_model_path = os.path.join(save_model_path, "pipe_model.pkl")
    ar_transformer_path = os.path.join(save_model_path, "ar_transformer.pkl")
    params_path = os.path.join(save_model_path, "hyperparams.json")
    metrics_path = os.path.join(save_model_path, "metrics.json")
    logging.info(
        f"Pipeline, Params, Metrics and AR transformer are saved in {save_model_path}:\n"
        f"{pipe_model_path}\n"
        f"{params_path}\n"
        f"{ar_transformer_path}"
    )
    joblib.dump(report["pipe_model"], pipe_model_path)
    joblib.dump(report["ar_transformer"], ar_transformer_path)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(report["params"], f, indent=4, ensure_ascii=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report["metrics"], f, indent=4, ensure_ascii=False)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    yt = np.array(pd.Series(y_true).to_numpy(copy=False),
                  dtype=np.float64, copy=True, order="C")
    yp = np.array(pd.Series(y_pred).to_numpy(copy=False),
                  dtype=np.float64, copy=True, order="C")
    return {
        "R2": r2_score(yt, yp),
        "RMSE": root_mean_squared_error(yt, yp),
        "MAE": mean_absolute_error(yt, yp),
    }
