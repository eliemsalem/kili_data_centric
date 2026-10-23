"""Étape 1 : benchmark (jeu fourni tel quel) et borne haute (étiquettes IMDB, mesure seulement)."""
from common import load_train, score, D
import pandas as pd

train = load_train()
m = pd.read_csv(f'{D}/match_imdb.csv', index_col=0)
print('benchmark tel quel      : %.4f ± %.4f' % score(train))

oracle = train.copy()
known = (m.sim > 0.5) & (m.split != 'unsupervised')
oracle.loc[known, 'category'] = m.loc[known, 'imdb_label']
print('borne haute (IMDB)      : %.4f ± %.4f' % score(oracle))
print('borne haute, IMDB seul  : %.4f ± %.4f' % score(oracle[known]))
