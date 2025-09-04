import os
import logging
import pandas as pd
import pytz
import joblib
from src.common.modeling_util import train_timeseries_model

SITE_TEST = {
    ('Totem 73 boulevard de Sébastopol', 'N-S'): {
        "save_sub_dir": "Sebastopol_N-S",
        "input_file_name": "initial_Sebastopol_N-S_with_feats.csv",
        "range": (0.0, 100.0),  # a portion of the original range TODO : use exact timestamp ?
        "temp_feats": [7, 1, 24],
        "test_ratio": 0.2,
        "iter_grid_search": 0,
    }
}

# -------------------------------------------------------------------
# Configuration des logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "train_and_predict.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Script principal
# -------------------------------------------------------------------
def main():
    '''
    This script train a XGBoost model based on the feature engineered dataset
    for this particular counter (a full df including train data and prediction data)
    Arguments:
        None (from dictionnary directly)
    Returns:
        exit 1 if error during traning and prediction
        exit 0 if OK
    '''
    for counter_id in SITE_TEST.keys():
        # working paths
        input_file_name = SITE_TEST[counter_id]["input_file_name"]
        # working paths
        input_file_path = os.path.join("data", "processed", input_file_name)
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # data load
        logging.info(f"Processed data load from [{input_file_path}]")
        df_counter = pd.read_csv(
            input_file_path,
            index_col=0,
        )
        # date format conversions
        df_counter["date_et_heure_de_comptage_utc"] = pd.to_datetime(
            df_counter["date_et_heure_de_comptage_utc"],
            format="%Y-%m-%d %H:%M:%S%z",
            utc=True
        )
        df_counter["date_et_heure_de_comptage_local"] = pd.to_datetime(
            df_counter["date_et_heure_de_comptage_local"],
            format="%Y-%m-%d %H:%M:%S%z",
            utc=True
        ).dt.tz_convert(pytz.timezone("Europe/Paris"))

        # train and predict on feature engineered processed data
        logging.info("Start training and gridsearch")
        report = train_timeseries_model(
            df_counter,
            target_col="comptage_horaire",
            timestamp_cols=["date_et_heure_de_comptage_local", "date_et_heure_de_comptage_utc"],
            temp_feats=SITE_TEST[counter_id]["temp_feats"],
            test_ratio=SITE_TEST[counter_id]["test_ratio"],
            iter_grid_search=SITE_TEST[counter_id]["iter_grid_search"],
        )

        # save final training/test/predictions data from the results :
        save_data_path = os.path.join("data", "final", SITE_TEST[counter_id]["save_sub_dir"])
        os.makedirs(save_data_path, exist_ok=True)
        X_train_path = os.path.join(save_data_path, "X_train.csv")
        X_test_path = os.path.join(save_data_path, "X_test.csv")
        X_train_dates_path = os.path.join(save_data_path, "X_train_dates.csv")
        X_test_dates_path = os.path.join(save_data_path, "X_test_dates.csv")
        y_train_path = os.path.join(save_data_path, "y_train.csv")
        y_test_path = os.path.join(save_data_path, "y_test.csv")
        y_train_pred_path = os.path.join(save_data_path, "y_train_pred.csv")
        y_test_pred_path = os.path.join(save_data_path, "y_test_pred.csv")
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

        # save pipeline model/params/transformer from the result
        save_model_path = os.path.join("models", SITE_TEST[counter_id]["save_sub_dir"])
        os.makedirs(save_model_path, exist_ok=True)
        model_path = os.path.join(save_model_path, "pipe_model.pkl")
        params_path = os.path.join(save_model_path, "params.pkl")
        ar_transformer_path = os.path.join(save_model_path, "ar_transformer.pkl")
        logging.info(
            f"Pipeline, Params and AR transformer PKL dumps saved in {save_data_path}:\n"
            f"{model_path}\n"
            f"{params_path}\n"
            f"{ar_transformer_path}"
        )
        joblib.dump(report["pipe_model"], model_path)
        joblib.dump(report["params"], params_path)
        joblib.dump(report["ar_transformer"], ar_transformer_path)

        logging.info("✅ Training and forecasting ended successfully.")
        exit(0)


if __name__ == "__main__":
    main()
