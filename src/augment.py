"""Étape 3d (sans données externes) : doubler le jeu avec nos propres critiques."""
import re
import numpy as np
import pandas as pd
from common import load_train, score, D

train = load_train()
y = train.category.values
p = np.load(f'{D}/oof_roberta-base.npy')
sus = np.where(y == 1, 1 - p, p) > 0.6
base = train.copy()
base.loc[sus, 'category'] = 1 - y[sus]


def norm(s):
    """Typographie IMDB : ' recollé, ponctuation et tirets détachés réparés."""
    s = re.sub(r" ' ", "'", s)
    s = re.sub(r" ([.,!?;:)])", r"\1", s)
    s = re.sub(r"\( ", "(", s)
    return s.replace(' - ', '-').replace('<br /> <br />', '<br /><br />')


normed = base.assign(review=base.review.map(norm))
print('base                  : %.4f ± %.4f' % score(base), flush=True)
print('normalisé seul        : %.4f ± %.4f' % score(normed), flush=True)
print('x2 duplication exacte : %.4f ± %.4f' % score(pd.concat([base, base])), flush=True)
print('x2 base + normalisé   : %.4f ± %.4f' % score(pd.concat([base, normed])), flush=True)
print('x2 normalisé          : %.4f ± %.4f' % score(pd.concat([normed, normed])), flush=True)
