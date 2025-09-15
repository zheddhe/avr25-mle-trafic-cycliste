# Manual call example

```bash
python -m src.ml.models.train_and_predict --processed-path data/processed/Sebastopol_N-S_v3/initial_with_feats.csv --target-col comptage_horaire --ts-col-utc date_et_heure_de_comptage_utc --ts-col-local date_et_heure_de_comptage_local --ar 7 --mm 1 --roll 24 --test-ratio 0.25 --grid-iter 0
```
