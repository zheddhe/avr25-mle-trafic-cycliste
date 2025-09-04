import pandas as pd
import argparse
import logging
from pathlib import Path
import os

# --- Configuration du logger ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocessing.log"),
        logging.StreamHandler(),
    ],
)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features à partir de la colonne de temps, en excluant les features temporelles (autorégressives/moyennes mobiles).
    """
    logging.info("Création des features...")
    # Correction pour gérer les informations de fuseau horaire
    df["date"] = pd.to_datetime(df["date_et_heure_de_comptage"], utc=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["hour"] = df["date"].dt.hour
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    
    logging.info("Features créées avec succès (sans variables autorégressives ni moyennes mobiles).")
    return df

def preprocess_data(input_file: Path, output_dir: Path, nom_compteur: str, data_type: str, iteration: int = None):
    """
    Lit un fichier de données brutes, le prétraite et le sauvegarde.
    """
    logging.info(f"Début du prétraitement pour le fichier: {input_file.name}")
    
    try:
        # Lire les données
        df = pd.read_csv(input_file, parse_dates=["date_et_heure_de_comptage"])
        
        # Simuler une fenêtre mobile de 75%
        df_initial = df.sample(frac=0.75, random_state=42)

        # Création des features
        df_with_feats = create_features(df_initial)
        
        # Définir le chemin de sortie
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enregistrer les données brutes de la fenêtre
        base_filename = f"{data_type}_{nom_compteur}"
        if data_type == "daily" and iteration is not None:
            base_filename = f"{data_type}_{iteration}_{nom_compteur}"

        output_path_raw = output_dir / f"{base_filename}.csv"
        df_initial.to_csv(output_path_raw, index=False)
        logging.info(f"Fichier initial enregistré dans: {output_path_raw}")

        # Enregistrer les données enrichies
        output_path_feats = output_dir / f"{base_filename}_with_feats.csv"
        df_with_feats.to_csv(output_path_feats, index=False)
        logging.info(f"Fichier enrichi enregistré dans: {output_path_feats}")
        
        logging.info("Prétraitement terminé avec succès.")
    
    except Exception as e:
        logging.error(f"Une erreur est survenue lors du prétraitement: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de prétraitement des données de trafic cycliste.")
    parser.add_argument("input_path", type=Path, help="Chemin vers le fichier de données brutes.")
    parser.add_argument("nom_compteur", type=str, help="Nom du compteur pour le nommage des fichiers.")
    parser.add_argument("data_type", choices=["initial", "daily"], help="Type de données à traiter: 'initial' ou 'daily'.")
    parser.add_argument("--iteration", type=int, help="Numéro d'itération pour les données journalières.")

    args = parser.parse_args()

    # Définir le chemin de sortie
    PROCESSED_DATA_DIR = Path("data/processed")

    # Exécuter le script
    preprocess_data(
        input_file=args.input_path,
        output_dir=PROCESSED_DATA_DIR,
        nom_compteur=args.nom_compteur,
        data_type=args.data_type,
        iteration=args.iteration
    )