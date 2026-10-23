"""Télécharge IMDB (HF stanfordnlp/imdb) dans data/imdb/ : référence pour l'appariement et l'évaluation."""
import os
from datasets import load_dataset

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'imdb')
os.makedirs(D, exist_ok=True)
for split, ds in load_dataset('stanfordnlp/imdb').items():
    ds.to_parquet(f'{D}/{split}.parquet')
    print(split, len(ds))
