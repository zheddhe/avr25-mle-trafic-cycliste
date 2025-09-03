import os
import logging
import pandas as pd
from src.common.preprocessing_util import train_test_split_time_aware

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
    This script extract the raw information based on the SITE_TEST dictionnary
    for this particular counter and in between the mentionned range
    Arguments:
        None (from dictionnary directly)
    Returns:
        exit 1 if error during processing
        exit 0 if OK
    '''
    # working paths
    raw_path = os.path.join("data", "interim",
                            "df_compteur.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # data load
    logging.info(f"Chargement des données brutes depuis {raw_path}")
    df = pd.read_csv(raw_path, index_col=0)

    # split train/test and features/dates/target
    logging.info("Découpage train/test en cours (75%/25%)")
    X_train, X_train_dates, X_test, X_test_dates, y_train, y_test = train_test_split_time_aware(
        df, ["date_et_heure_de_comptage"], "comptage_horaire", test_size=0.2
    )

    # save the processed data
    logging.info("Sauvegarde des fichiers dans data/processed")
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=True)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=True)
    X_train_dates.to_csv(os.path.join(processed_dir, "X_train_dates.csv"), index=True, header=True)
    X_test_dates.to_csv(os.path.join(processed_dir, "X_test_dates.csv"), index=True, header=True)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=True, header=True)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=True, header=True)

    logging.info("✅ Dataset pour le machine learning extrait avec succès.")
    exit(0)


if __name__ == "__main__":
    main()
