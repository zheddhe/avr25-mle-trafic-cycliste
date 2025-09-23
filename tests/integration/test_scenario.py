RAW_FILE_NAME = "comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv"

SITE_TEST = {
    ('Totem 73 boulevard de Sébastopol', 'N-S', 'initial'): {
        "sub_dir": "Sebastopol_N-S_integration",
        "interim_file_name": "initial.csv",
        "processed_file_name": "initial_with_feats.csv",
        "range": (0.0, 75.0),
        "temp_feats": [7, 1, 24],
        "test_ratio": 0.25,
        "iter_grid_search": 0,
    },
    # ('Totem 73 boulevard de Sébastopol', 'N-S', 'day1'): {
    #     "sub_dir": "Sebastopol_N-S_integration_day1",
    #     "interim_file_name": "daily_1.csv",
    #     "processed_file_name": "daily_1_with_feats.csv",
    #     "range": (0.5, 75.5),
    #     "temp_feats": [7, 1, 24],
    #     "test_ratio": 0.25,
    #     "iter_grid_search": 0,
    # },
    # ('Totem 73 boulevard de Sébastopol', 'N-S', 'day2'): {
    #     "sub_dir": "Sebastopol_N-S_integration_day2",
    #     "interim_file_name": "daily_2.csv",
    #     "processed_file_name": "daily_2_with_feats.csv",
    #     "range": (1.0, 76.0),
    #     "temp_feats": [7, 1, 24],
    #     "test_ratio": 0.25,
    #     "iter_grid_search": 0,
    # },
}
