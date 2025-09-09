import os
import logging
import pandas as pd
from utils import DatetimePeriodicsTransformer

SITE_TEST = {
    ('Totem 73 boulevard de Sébastopol', 'N-S'): {
        "input_file_name": "initial_Sebastopol_N-S.csv",
        "processed_file_name": "initial_Sebastopol_N-S_with_feats.csv",
        "short_name": "Sebastopol_N-S",
        "range": (0.0, 75.0),  # a portion of the original range TODO : use exact timestamp ?
    }
}

COLUMNS_TO_DROP = [
    "nom_du_site_de_comptage",
    "orientation_compteur",
    "weather_code_wmo_code",
    "temperature_2m_c",
    "rain_mm",
    "snowfall_cm",
    "weather_code_wmo_code_category",
    "latitude",
    "longitude",
    "arrondissement",
    "elevation",
    "date_et_heure_de_comptage_hour",
    "date_et_heure_de_comptage_day",
    "date_et_heure_de_comptage_day_of_year",
    "date_et_heure_de_comptage_day_of_week",
    "date_et_heure_de_comptage_week",
    "date_et_heure_de_comptage_month",
    "date_et_heure_de_comptage_year",
    "date_et_heure_de_comptage_sin_week",
    "date_et_heure_de_comptage_cos_week",
    "date_et_heure_de_comptage_cos_day_of_year",
    "date_et_heure_de_comptage_sin_day_of_year",
]

# -------------------------------------------------------------------
# Configuration des logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "build_features.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Script principal
# -------------------------------------------------------------------
def main():
    '''
    This script construct the expected features and apply a drop unwanted columns

    Arguments:
        None (from dictionnary of configuration)

    Returns:
        exit 1 if error during feature engineering refinement
        exit 0 if OK
    '''
    for counter_id in SITE_TEST.keys():
        # working paths
        input_file_name = SITE_TEST[counter_id]["input_file_name"]
        input_file_path = os.path.join("data", "processed", input_file_name)
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # data load
        logging.info(f"Interim data load for counter [{counter_id}] from [{input_file_path}]")
        df = pd.read_csv(input_file_path, index_col=0)

        # enrich periodic features
        logging.info("Processing periodic temporal data feature engineering")
        timestamp_col = "date_et_heure_de_comptage"
        tr_date = DatetimePeriodicsTransformer(timestamp_col)
        df = tr_date.transform(df)

        # filter unwanted features
        logging.info(f"Filtering columns to drop : [{COLUMNS_TO_DROP}]")
        df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])

        # save the processed data
        processed_file_name = SITE_TEST[counter_id]["processed_file_name"]
        logging.info(f"Saving file [{processed_file_name}] at path [{processed_dir}]")
        df.to_csv(os.path.join(processed_dir, processed_file_name), index=True)

        logging.info("✅ Feature engineering processed successfully.")
        exit(0)


if __name__ == "__main__":
    main()
