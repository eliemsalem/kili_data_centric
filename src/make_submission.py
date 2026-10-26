"""Builds the submitted set: roberta label flipping (t=0.6) + a normalised copy = 20,000 rows."""
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

back = pd.read_csv(path, index_col=0)  # re-read: what Kili will read
assert back.shape == sub.shape and (back.review == sub.review).all() and (back.category == sub.category).all()
acc, sd = score(back)
print(f'{path} : {len(back)} rows, {back.category.mean():.3f} positive, local score {acc:.4f} ± {sd:.4f}')
reg = f'{ROOT}/submissions/registry.csv'
row = pd.DataFrame([{'date': date.today().isoformat(), 'file': f'{name}.csv', 'rows': len(back),
                     'settings': 'roberta-base OOF flipping t=0.6 + normalised copy (x2)',
                     'local_score': round(acc, 4), 'platform_score': ''}])
row.to_csv(reg, mode='a', header=not pd.io.common.file_exists(reg), index=False)
