# Retail Product Visual Search and Recommendation Engine

CSAI3027P Deep Learning, individual hackathon. Aditya Dagar, SAP ID 590015198, B.Tech CSE 2nd Year, UPES Dehradun.

Upload a product photo, get visually similar catalogue items with similarity scores.
Not intended for identifying people, surveillance, or high-stakes decisions.

## Layout
- `Phase_I/`: proposal, dataset card, baselines, curves, architecture diagram
- `Phase_II/`: ablation, error analysis, app screenshots, model card
- `src/`: data, models, metrics, training, serving code
- `notebooks/`: Kaggle notebooks (exported)
- `manifests/`: `split_A.csv` (seen categories), `split_B.csv` (unseen categories)
- `configs/`: all hyperparameters
- `source_log.md`, `experiment_log.md`: disclosure and run history

## Reproduce
```
pip install -r requirements.txt
python src/utils.py   # prints environment info
```
Seed 42 everywhere. Details are filled in per phase.
