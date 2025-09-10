import os
import logging
import pandas as pd
import pytz
from models_utils import (
    train_timeseries_model,
    save_artefacts,
)
from mlflow_tracking import (
    configure_mlflow_from_env,
    start_run,
    log_report_content,
    log_model_with_signature,
    log_local_artifacts,
)
from src.ml.test_config import SITE_TEST

# -------------------------------------------------------------------
# Configuration des logs
# -------------------------------------------------------------------
log_dir = os.path.join("logs", "ml")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "train_and_predict.log")

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
    for this particular counter (which is a full dataframe including train data
    and prediction data)
    It stores the result of the training process (train/test split time aware)
    along with the predictions and finally the custom metrics for train and test
    in a json file along with intermediate split/enriched data used during the process

    Arguments:
        None (from dictionnary directly)

    Returns:
        exit 1 if error during traning and prediction
        exit 0 if OK
    '''
    # set MLflow tracking URI from env if provided
    configure_mlflow_from_env()

    for counter_id in SITE_TEST.keys():
        # working paths
        sub_dir = SITE_TEST[counter_id]["sub_dir"]
        processed_file_name = SITE_TEST[counter_id]["processed_file_name"]
        data_dir = os.path.join("data", "processed", sub_dir)
        model_dir = os.path.join("models", sub_dir)
        processed_file_path = os.path.join(data_dir, processed_file_name)
        os.makedirs(model_dir, exist_ok=True)

        # data load
        logging.info(f"Processed data load from [{processed_file_path}]")
        df_counter = pd.read_csv(
            processed_file_path,
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

        # save all artefacts with standard filenames in specific counter subdir
        tags = {
            "counter.name": str(counter_id[0]),
            "counter.orientation": str(counter_id[1]),
            "model.family": "XGBRegressor",
        }
        with start_run(experiment_name=sub_dir, run_name=processed_file_name, tags=tags):
            # log report content (metrics, shapes, params)
            log_report_content(report, target_col="comptage_horaire")
            # persist artefacts locally (unchanged)
            save_artefacts(report, sub_dir)
            # log model with signature (use one real train row)
            x_sample = report["X_train"].head(1)
            log_model_with_signature(
                pipe_model=report["pipe_model"],
                sample_input_df=x_sample,
                artifact_path="model_pipeline",
                registered_name=f"{sub_dir}-model"
            )
            # log all produced files/dirs as MLflow artifacts
            log_local_artifacts(sub_dir)

    logging.info("✅ Training and forecasting ended successfully.")
    exit(0)


if __name__ == "__main__":
    main()
