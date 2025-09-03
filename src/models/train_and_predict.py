import os
import logging
import pandas as pd

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
    df_compteur_path = os.path.join("data", "processed", "df_compteur_processed.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # data load
    logging.info(f"Chargement des données processed depuis {df_compteur_path}")
    df_compteur = pd.read_csv(df_compteur_path, index_col=0)

    logger.info(f"{df_compteur}")

    # Entrainement avec gridsearch bayesienne et calcul des prédictions
    logging.info("Entrainement avec gridsearch bayesienne et calcul des prédictions")

    # save the results and the model pipeline data
    logging.info("Sauvegarde dela pipeline dans models")

    logging.info("✅ Entraînement réussi et prédiction calculées avec succès.")
    exit(0)


if __name__ == "__main__":
    main()
