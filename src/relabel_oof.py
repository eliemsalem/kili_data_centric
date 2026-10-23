"""Étape 3b : étiquettes suspectes par prédictions out-of-fold (TF-IDF + régression logistique)."""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import load_train, score, D

train = load_train()
m = pd.read_csv(f'{D}/match_imdb.csv', index_col=0)
known = (m.sim > 0.5) & (m.split != 'unsupervised')
truly_wrong = (train.category != m.imdb_label)[known]

X = hstack([TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(train.review),
            TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                            max_features=300_000).fit_transform(train.review)]).tocsr()
y = train.category.values
p = np.zeros(len(y))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    p[va] = LogisticRegression(C=4, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
np.save(f'{D}/oof_lr.npy', p)
p_other = np.where(y == 1, 1 - p, p)  # probabilité de l'autre étiquette
print('accord OOF vs IMDB (vraie étiquette) :', ((p > .5) == m.imdb_label)[known].mean())

os.environ['FT_QUIET'] = '1'
for t in [0.5, 0.6, 0.7, 0.8]:
    sus = pd.Series(p_other > t, index=train.index)
    prec = truly_wrong[sus[known]].mean()
    rec = sus[known][truly_wrong].mean()
    flip = train.copy()
    flip.loc[sus, 'category'] = 1 - flip.loc[sus, 'category']
    print(f't={t} suspects={sus.sum()} précision={prec:.3f} rappel={rec:.3f} | '
          'inverser %.4f ± %.4f' % score(flip), '| supprimer %.4f ± %.4f' % score(train[~sus]), flush=True)
