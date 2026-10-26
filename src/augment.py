"""Step 3d (no external data): double the training set with our own reviews."""
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
    """IMDB typography: rejoins ' and reattaches detached punctuation and hyphens."""
    s = re.sub(r" ' ", "'", s)
    s = re.sub(r" ([.,!?;:)])", r"\1", s)
    s = re.sub(r"\( ", "(", s)
    return s.replace(' - ', '-').replace('<br /> <br />', '<br /><br />')


normed = base.assign(review=base.review.map(norm))
print('base                  : %.4f ± %.4f' % score(base), flush=True)
print('normalised only       : %.4f ± %.4f' % score(normed), flush=True)
print('x2 exact duplicate    : %.4f ± %.4f' % score(pd.concat([base, base])), flush=True)
print('x2 base + normalised  : %.4f ± %.4f' % score(pd.concat([base, normed])), flush=True)
print('x2 normalised         : %.4f ± %.4f' % score(pd.concat([normed, normed])), flush=True)
