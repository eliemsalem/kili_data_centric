"""Matches each Kili training review to its closest IMDB source review (word TF-IDF, cosine)."""
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def norm(s):
    s = s.lower().replace('<br />', ' ')
    s = re.sub(r"\s*'\s*", "'", s)
    return re.sub(r'\s+', ' ', s)


train = pd.read_csv(f'{D}/full_train.csv', index_col=0)
imdb = pd.concat([pd.read_parquet(f'{D}/imdb/{s}.parquet').assign(split=s)
                  for s in ['train', 'test', 'unsupervised']], ignore_index=True)

vec = TfidfVectorizer(token_pattern=r"[a-z']{2,}", sublinear_tf=True, min_df=2, dtype=np.float32)
B = vec.fit_transform(imdb.text.map(norm))
A = vec.transform(train.review.map(norm))
best, sim = np.empty(A.shape[0], int), np.empty(A.shape[0], np.float32)
for i in range(0, A.shape[0], 500):
    S = (A[i:i + 500] @ B.T).toarray()
    best[i:i + 500] = S.argmax(1)
    sim[i:i + 500] = S.max(1)
m = imdb.iloc[best].reset_index(drop=True)
out = pd.DataFrame({'sim': sim, 'imdb_idx': best, 'split': m.split.values,
                    'imdb_label': m.label.values, 'category': train.category.values}, index=train.index)
out.to_csv(f'{D}/match_imdb.csv')
print(out.sim.describe())
for t in [0.3, 0.5, 0.7]:
    print(t, (out.sim > t).mean())
ok = out[out.sim > 0.5]
print(ok.split.value_counts())
lab = ok[ok.split != 'unsupervised']
print('Kili vs IMDB label agreement:', (lab.category == lab.imdb_label).mean(), 'n=', len(lab))
print('distinct IMDB sources:', ok.imdb_idx.nunique(), '/', len(ok))
