import os
import logging
import pandas as pd
from data_utils import apply_percent_range_selection

RAW_FILE_NAME = "comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv"

SITE_TEST = {
    ('Totem 73 boulevard de Sébastopol', 'N-S'): {
        "sub_dir": "Sebastopol_N-S",
        "output_file_name": "initial.csv",
        "range": (0.0, 76.0),  # a portion of the original range TODO : use exact timestamp ?
    },
    # ('Totem 73 boulevard de Sébastopol', 'N-S'): {
    #     "sub_dir": "Sebastopol_N-S",
    #     "output_file_name": "daily_1.csv",
    #     "range": (0.1, 75.1),  # a portion of the original range TODO : use exact timestamp ?
    # }
}

# -------------------------------------------------------------------
# Logs configuration
# -------------------------------------------------------------------
log_dir = os.path.join("logs", "ml")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "import_raw_data.log")

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
    raw_data_path = os.path.join("data", "raw", RAW_FILE_NAME)

    # data load
    logger.info(f"Raw data load from {raw_data_path}")
    df = pd.read_csv(raw_data_path, index_col=0)

    # extract data for choosen counter
    logger.info(f"Context [{SITE_TEST}]")
    grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])
    counter_found = False
    for counter_id, df_counter in grouped:
        if counter_id in SITE_TEST:
            output_dir = os.path.join("data", "interim", SITE_TEST[counter_id]["sub_dir"])
            os.makedirs(output_dir, exist_ok=True)
            df = df_counter.copy()
            logger.info(f"Counter [{counter_id}] found")
            # convert date column, sort by date and reindex the file
            logger.info("Sorting the counter data by date")
            timestamp_col = "date_et_heure_de_comptage"
            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col],
                format="%Y-%m-%dT%H:%M:%S%z",
                utc=True
            )
            df = df.sort_values(timestamp_col).reset_index().drop(columns=["index"])
            # restrict the data to the selected range (simulate production data collection)
            logger.info(
                f"Range for this counter [{SITE_TEST[counter_id]["range"]} (in percent)]"
            )
            df = apply_percent_range_selection(
                df,
                SITE_TEST[counter_id]["range"],
            )
            # save intermediate data
            output_file_name = SITE_TEST[counter_id]["output_file_name"]
            logger.info(f"Saving file [{output_file_name}] at path [{output_dir}]")
            df.to_csv(
                os.path.join(output_dir, output_file_name),
                index=True
            )
            counter_found = True
            break
    if not counter_found:
        logger.warning("⚠️ Raw counters data extraction hasn't detected any counter.")
        exit(1)

    logger.info("✅ Raw counters data extraction is successfull.")
    exit(0)


if __name__ == "__main__":
    main()
