import pandas as pd
import numpy as np
import os
import logging
import sys

# Configuration du logger pour le projet
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Charge les données à partir d'un fichier CSV en utilisant Pandas."""
    try:
        logger.info(f"Chargement des données depuis {file_path}...")
        df = pd.read_csv(file_path)
        logger.info("Données chargées avec succès.")
        return df
    except FileNotFoundError:
        logger.error(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        sys.exit(1)

def filter_and_select(df):
    """Filtre les données selon les critères du projet (75% des compteurs, sans données manquantes et fort trafic)."""
    logger.info("Début du filtrage des données...")
    
    # Étape 1 : Supprimer les lignes avec des valeurs manquantes
    df_cleaned = df.dropna()
    logger.info(f"Nombre de lignes après suppression des données manquantes : {len(df_cleaned)}")

    # Étape 2 : Identifier les compteurs avec un fort trafic
    traffic_per_counter = df_cleaned.groupby('nom_du_site_de_comptage')['comptage_horaire'].sum()
    high_traffic_counters = traffic_per_counter[traffic_per_counter > traffic_per_counter.median()].index
    df_filtered = df_cleaned[df_cleaned['nom_du_site_de_comptage'].isin(high_traffic_counters)]
    
    # Étape 3 : Sélectionner 75% des compteurs filtrés de manière aléatoire
    all_counters = df_filtered['nom_du_site_de_comptage'].unique()
    num_to_select = int(len(all_counters) * 0.75)
    selected_counters = np.random.choice(all_counters, size=num_to_select, replace=False)
    
    df_final = df_filtered[df_filtered['nom_du_site_de_comptage'].isin(selected_counters)]
    
    logger.info(f"Filtrage terminé. {len(selected_counters)} compteurs sur {len(all_counters)} ont été sélectionnés.")
    return df_final

def split_data_time_aware(df):
    """Divise les données en ensembles d'entraînement et de test basés sur le temps."""
    logger.info("Début de la séparation des données en X et y (time-aware)...")
    
    # Assurez-vous que la colonne de date existe.
    df['date_et_heure_de_comptage'] = pd.to_datetime(df['date_et_heure_de_comptage'])
    df = df.sort_values(by='date_et_heure_de_comptage')
    
    # Séparation des données
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    
    # Définir les features (X) et la cible (y)
    features = [col for col in df.columns if col not in ['comptage_horaire', 'nom_du_site_de_comptage', 'date_et_heure_de_comptage']]
    
    X_train = train_df[features]
    y_train = train_df['comptage_horaire']
    X_test = test_df[features]
    y_test = test_df['comptage_horaire']
    
    logger.info("Séparation en ensembles d'entraînement et de test (time-aware) effectuée.")
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    """Sauvegarde les ensembles de données dans le répertoire processed."""
    logger.info("Sauvegarde des données dans data/processed...")
    os.makedirs('data/processed', exist_ok=True)
    
    X_train.to_csv('data/processed/X_train.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    logger.info("Données sauvegardées avec succès.")

def save_interim_data(df):
    """Sauvegarde le fichier de données intermédiaires."""
    interim_path = 'data/interim/'
    os.makedirs(interim_path, exist_ok=True)
    file_path = os.path.join(interim_path, 'daily_initial.csv')
    df.to_csv(file_path, index=False)
    logger.info(f"Fichier intermédiaire sauvegardé dans {file_path}")

if __name__ == "__main__":
    # Chemin d'accès au fichier de données brutes
    raw_data_path = 'data/raw/comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv'
    
    # Étape 1 : Chargement des données brutes
    raw_data = load_data(raw_data_path)
    
    # Étape 2 : Filtrage et sélection
    processed_data = filter_and_select(raw_data)
    
    # Étape 3 : Sauvegarde des données intermédiaires
    save_interim_data(processed_data)
    
    # Étape 4 : Séparation en jeux de données d'entraînement et de test (time-aware)
    X_train, X_test, y_train, y_test = split_data_time_aware(processed_data)
    
    # Étape 5 : Sauvegarde des jeux de données finaux
    save_data(X_train, X_test, y_train, y_test)