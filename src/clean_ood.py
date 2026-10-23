"""Étape 3a : effet du retrait du hors domaine et des doublons, après inversion roberta (t=0,6)."""
import numpy as np
import pandas as pd
from common import load_train, score, D

train = load_train()
y = train.category.values
m = pd.read_csv(f'{D}/match_imdb.csv', index_col=0)
p = np.load(f'{D}/oof_roberta-base.npy')
sus = np.where(y == 1, 1 - p, p) > 0.6
base = train.copy()
base.loc[sus, 'category'] = 1 - y[sus]

nonascii = train.review.map(lambda s: sum(ord(c) > 127 for c in s) / len(s)).values
print('base (inversion roberta)            : %.4f ± %.4f' % score(base), flush=True)
print('- doublons exacts                   : %.4f ± %.4f' % score(base[~train.review.duplicated()]), flush=True)
print('- non-ASCII > 20 %% (%d)             : %.4f ± %.4f' % ((nonascii > .2).sum(), *score(base[nonascii <= .2])), flush=True)
print('- sim IMDB < 0,5 (%d, borne haute)  : %.4f ± %.4f' % ((m.sim < .5).sum(), *score(base[(m.sim >= .5).values])), flush=True)
