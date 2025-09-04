import os
import logging
import pandas as pd
from src.common.preprocessing_util import apply_percent_range_selection

SITE_TEST = {
    ('Totem 73 boulevard de Sébastopol', 'N-S'): {
        "short_name": "Sebastopol_N-S",
        "range": (0.0, 75.0),  # a portion of the original range TODO : use exact timestamp ?
    }
}

# -------------------------------------------------------------------
# Logs configuration
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
# Main script
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
    raw_path = os.path.join(
        "data", "raw",
        "comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv"
    )
    interim_dir = os.path.join("data", "interim")
    os.makedirs(interim_dir, exist_ok=True)

    # data load
    logger.info(f"Raw data load from {raw_path}")
    df = pd.read_csv(raw_path, index_col=0)

    # extract data for choosen counter
    logger.info(f"Context [{SITE_TEST}]")
    grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])
    counter_found = False
    for counter_id, df_counter in grouped:
        if counter_id in SITE_TEST:
            logger.info(f"Counter [{counter_id}] found")
            logger.info(f"Range for this counter [{SITE_TEST[counter_id]["range"]}(percentile)]")
            df_counter = apply_percent_range_selection(
                df_counter,
                SITE_TEST[counter_id]["range"],
            )
            # save extracted data
            save_file_name = f"df_{SITE_TEST[counter_id]["short_name"]}.csv"
            logger.info(f"Saving dile [{save_file_name}] at path [{interim_dir}]")
            df_counter.to_csv(
                os.path.join(interim_dir, save_file_name),
                index=True
            )
            counter_found = True
            break
    if counter_found:
        logger.info("✅ Raw counters data extraction is successfull.")
        exit(0)
    else:
        logger.warning("❌ Raw counters data extraction have failed.")
        exit(1)


if __name__ == "__main__":
    main()
