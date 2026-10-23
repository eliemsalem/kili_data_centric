"""Construit le jeu soumis : inversion roberta (t=0,6) + copie normalisée = 20 000 lignes."""
import sys
from datetime import date
import numpy as np
import pandas as pd
from augment import base, normed
from common import score, ROOT

name = sys.argv[1]
sub = pd.concat([base, normed], ignore_index=True)[['review', 'category']]
sub['review'] = sub.review.str.replace(r'[\r\n]+', ' ', regex=True)
sub['category'] = sub.category.astype(int)
assert len(sub) <= 20_000 and sub.notna().all().all() and set(sub.category) == {0, 1}
path = f'{ROOT}/submissions/{name}.csv'
sub.to_csv(path)

back = pd.read_csv(path, index_col=0)  # relecture : ce que Kili lira
assert back.shape == sub.shape and (back.review == sub.review).all() and (back.category == sub.category).all()
acc, sd = score(back)
print(f'{path} : {len(back)} lignes, {back.category.mean():.3f} de positifs, score local {acc:.4f} ± {sd:.4f}')
reg = f'{ROOT}/submissions/registre.csv'
row = pd.DataFrame([{'date': date.today().isoformat(), 'fichier': f'{name}.csv', 'lignes': len(back),
                     'réglages': 'inversion roberta-base OOF t=0.6 + copie normalisée (x2)',
                     'score_local': round(acc, 4), 'score_plateforme': ''}])
row.to_csv(reg, mode='a', header=not pd.io.common.file_exists(reg), index=False)
