"""Local evaluation: Kili's fixed fastText script, on a clean IMDB set absent from the training data."""
import os
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{ROOT}/data'
sys.path.insert(0, f'{D}/kili_script')
import csv  # noqa: E402
import train as kili  # noqa: E402  (Kili's script, unchanged)
from train import train_model  # noqa: E402


def _process_csv(df, output_name, output_folder):
    # Copy of kili.process_csv: Python >= 3.11 rejects quotechar="", None gives the same output.
    df.loc[:, 'text'] = '__label__' + df['category'].astype(str) + ' ' + df['review']
    output_file = f'{output_folder}/{output_name}.txt'
    df[['text']].to_csv(output_file, index=False, header=False,
                        quoting=csv.QUOTE_NONE, quotechar=None, escapechar="\\")
    return output_file


kili.process_csv = _process_csv


def load_train():
    return pd.read_csv(f'{D}/full_train.csv', index_col=0)


def load_eval(n=10_000, seed=0):
    """IMDB reviews (labelled train+test) that match no row of the Kili training set."""
    path = f'{D}/eval_imdb.csv'
    if not os.path.exists(path):
        imdb = pd.concat([pd.read_parquet(f'{D}/imdb/{s}.parquet') for s in ['train', 'test']],
                         ignore_index=True)
        used = set(pd.read_csv(f'{D}/match_imdb.csv').imdb_idx)
        imdb = imdb[~imdb.index.isin(used)]
        ev = imdb.sample(n, random_state=seed).rename(columns={'text': 'review', 'label': 'category'})
        ev.to_csv(path)
    return pd.read_csv(path, index_col=0)


def score(df, df_eval=None, runs=3):
    """Mean accuracy and standard deviation of fastText (Kili's script) trained on df."""
    df_eval = load_eval() if df_eval is None else df_eval
    df = df[['review', 'category']].copy()
    df['review'] = df.review.str.replace(r'[\r\n]+', ' ', regex=True)
    accs = []
    for _ in range(runs):
        with tempfile.TemporaryDirectory() as tmp:
            # fastText writes its progress to the C stderr: silence it during training
            err = os.dup(2)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            try:
                accs.append(train_model(df.copy(), df_eval.copy(), tmp))
            finally:
                os.dup2(err, 2)
    return float(np.mean(accs)), float(np.std(accs))
