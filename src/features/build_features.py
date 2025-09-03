import os
import logging
import pandas as pd
from src.common.preprocessing_util import DatetimePeriodicsTransformer

COLUMNS_TO_DROP = [
    "weather_code_wmo_code",
    "date_et_heure_de_comptage_hour",
    "date_et_heure_de_comptage_day",
    "date_et_heure_de_comptage_day_of_year",
    "date_et_heure_de_comptage_day_of_week",
    "date_et_heure_de_comptage_week",
    "date_et_heure_de_comptage_month",
    "date_et_heure_de_comptage_year",
    "latitude",
    "longitude",
    "arrondissement",
    "elevation",
    "date_et_heure_de_comptage_sin_week",
    "date_et_heure_de_comptage_cos_week",
    "date_et_heure_de_comptage_cos_day_of_year",
    "date_et_heure_de_comptage_sin_day_of_year",
]

# -------------------------------------------------------------------
# Configuration des logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "make_dataset.log")

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
    This script construct the expected features and apply a customizable filtering
    of columns to drop
    Arguments:
        None (from dictionnary of configuration)
    Returns:
        exit 1 if error during feature engineering refinement
        exit 0 if OK
    '''
    # working paths
    interim_path = os.path.join("data", "interim",
                                "df_compteur.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # data load
    logging.info(f"Chargement des données intermédiaires du compteur depuis {interim_path}")
    df = pd.read_csv(interim_path, index_col=0)

    # enrich periodic features
    timestamp_col = "date_et_heure_de_comptage"
    tr_date = DatetimePeriodicsTransformer(timestamp_col)
    df = tr_date.transform(df)
    timestamp_col = timestamp_col+"_local"

    # filter unwanted features
    logging.info(f"Filtrage des features suivantes : {COLUMNS_TO_DROP}")
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])

    # save the processed data
    logging.info("Sauvegarde du fichier df_compteur_processed.csv dans data/processed")
    df.to_csv(os.path.join(processed_dir, "df_compteur_processed.csv"), index=True)

    logging.info("✅ Dataset pour le machine learning enrichi avec succès.")
    exit(0)


if __name__ == "__main__":
    main()
