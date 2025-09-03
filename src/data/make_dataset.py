import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Configuration des logs
# -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "split.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# -------------------------------------------------------------------
# Script principal
# -------------------------------------------------------------------
def main():
    # chemins
    raw_path = os.path.join("data", "raw", "raw.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # chargement des données
    logging.info(f"Chargement des données brutes depuis {raw_path}")
    df = pd.read_csv(raw_path)

    # séparation features (la date n'est pas considérée comme pertinente et mise de côté) / cible
    target_col = "silica_concentrate"
    date_col = "date"
    X = df.drop(columns=[target_col, date_col])
    y = df[target_col]

    # split train/test
    logging.info("Découpage train/test en cours (80%/20%)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # sauvegarde
    logging.info("Sauvegarde des fichiers dans data/processed")
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False, header=True)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False, header=True)

    logging.info("✅ Split terminé avec succès.")


if __name__ == "__main__":
    main()
