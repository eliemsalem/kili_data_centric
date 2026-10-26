"""Assembles a teacher's OOF folds and measures flipping the suspect labels (Kili's script)."""
import glob
import sys
import numpy as np
import pandas as pd
from common import load_train, score, D

prefix = sys.argv[1] if len(sys.argv) > 1 else 'roberta-base'
train = load_train()
y = train.category.values
m = pd.read_csv(f'{D}/match_imdb.csv', index_col=0)
known = ((m.sim > 0.5) & (m.split != 'unsupervised')).values
imdb = m.imdb_label.values

p = np.full(len(y), np.nan)
files = sorted(glob.glob(f'{D}/oof/{prefix}_f*.npz'))
for f in files:
    z = np.load(f)
    p[z['idx']] = z['p']
assert not np.isnan(p).any(), f'missing folds: {len(files)}/5'
np.save(f'{D}/oof_{prefix}.npy', p)

teachers = {prefix: p, f'{prefix}+lr': (p + np.load(f'{D}/oof_lr.npy')) / 2}
for name, q in teachers.items():
    print(f'== {name} : IMDB agreement {((q > .5) == imdb)[known].mean():.4f}', flush=True)
    p_other = np.where(y == 1, 1 - q, q)
    for t in [0.5, 0.6, 0.7]:
        sus = p_other > t
        flip = train.copy()
        flip.loc[sus, 'category'] = 1 - y[sus]
        prec = (y != imdb)[known & sus].mean()
        rec = sus[known & (y != imdb)].mean()
        print(f't={t} suspects={sus.sum()} precision={prec:.3f} recall={rec:.3f} | '
              'flip %.4f ± %.4f' % score(flip), flush=True)
