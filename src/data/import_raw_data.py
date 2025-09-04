import os
import logging
import pandas as pd
from src.common.preprocessing_util import apply_percent_range_selection

SITE_TEST = {
    ('Totem 73 boulevard de Sébastopol', 'N-S'): {
        "short_name": "Sebastopol_N-S",
        "range": (0.0, 100.0),  # a portion of the original range TODO : use exact timestamp ?
    }
}

# -------------------------------------------------------------------
# Configuration des logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "import_raw_data.log")

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
    raw_path = os.path.join("data", "raw",
                            "comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv")
    interim_dir = os.path.join("data", "interim")
    os.makedirs(interim_dir, exist_ok=True)

    # data load
    logger.info(f"Chargement des données brutes depuis {raw_path}")
    df = pd.read_csv(raw_path, index_col=0)

    # extract data for choosen counter
    logger.info(f"récupération des données de comptage avec filtrage : {SITE_TEST}")
    grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])
    compteur_trouve = False
    for compteur_id, df_compteur in grouped:
        logger.info(f"compteur : {compteur_id}")
        if compteur_id in SITE_TEST:
            logger.info(f"range conservé : {SITE_TEST[compteur_id]["range"]}")
            df_compteur = apply_percent_range_selection(
                df_compteur,
                SITE_TEST[compteur_id]["range"],
            )
            # save extracted data
            logger.info("Sauvegarde du fichier df_compteur.csv dans data/interim")
            df_compteur.to_csv(os.path.join(interim_dir, "df_compteur.csv"),
                               index=True)
            compteur_trouve = True
            break
    if compteur_trouve:
        logger.info("✅ Extraction des données de compteur terminé avec succès.")
        exit(0)
    else:
        logger.warning("❌ ImpExtraction des données de compteur en échec.")
        exit(1)


if __name__ == "__main__":
    main()
