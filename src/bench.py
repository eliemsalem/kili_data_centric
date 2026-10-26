"""Step 1: benchmark (provided set as is) and upper bound (IMDB labels, measurement only)."""
from common import load_train, score, D
import pandas as pd

train = load_train()
m = pd.read_csv(f'{D}/match_imdb.csv', index_col=0)
print('benchmark as is        : %.4f ± %.4f' % score(train))

oracle = train.copy()
known = (m.sim > 0.5) & (m.split != 'unsupervised')
oracle.loc[known, 'category'] = m.loc[known, 'imdb_label']
print('upper bound (IMDB)     : %.4f ± %.4f' % score(oracle))
print('upper bound, IMDB only : %.4f ± %.4f' % score(oracle[known]))
