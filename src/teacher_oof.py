"""Professeur roberta-base affiné en OOF 5 plis sur les étiquettes bruitées (GTX 1080 Ti, fp32)."""
import argparse
import os
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)
from common import load_train, D

ap = argparse.ArgumentParser()
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--model', default='roberta-base')
ap.add_argument('--max_len', type=int, default=512)
ap.add_argument('--epochs', type=float, default=2)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--max_steps', type=int, default=-1)  # test de vitesse
args = ap.parse_args()

train = load_train()
texts = train.review.str.replace('<br />', ' ').tolist()
y = train.category.values
tr, va = list(StratifiedKFold(5, shuffle=True, random_state=0).split(texts, y))[args.fold]

tok = AutoTokenizer.from_pretrained(args.model)
enc = lambda idx: Dataset.from_dict({'text': [texts[i] for i in idx], 'label': y[idx]}).map(
    lambda b: tok(b['text'], truncation=True, max_length=args.max_len), batched=True, remove_columns=['text'])
ds_tr, ds_va = enc(tr), enc(va)

set_seed(args.fold)  # avant de créer le modèle (tête initialisée au hasard)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
name = f'{args.model.split("/")[-1]}_f{args.fold}'
targs = TrainingArguments(
    output_dir=f'{D}/../models/{name}', per_device_train_batch_size=8, per_device_eval_batch_size=32,
    gradient_accumulation_steps=2, learning_rate=args.lr, num_train_epochs=args.epochs,
    max_steps=args.max_steps, warmup_ratio=0.06, weight_decay=0.01, lr_scheduler_type='linear',
    group_by_length=True, save_strategy='no', logging_steps=50, report_to=[], fp16=False,
    dataloader_num_workers=2, seed=args.fold)
trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, data_collator=DataCollatorWithPadding(tok))
trainer.train()
if args.max_steps > 0:
    raise SystemExit
trainer.args.group_by_length = False  # cf. mémo Posos : ne jamais trier en prédiction
logits = trainer.predict(ds_va).predictions
p = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
os.makedirs(f'{D}/oof', exist_ok=True)
np.savez(f'{D}/oof/{name}.npz', idx=va, p=p)
print(f'pli {args.fold} : accuracy vs étiquettes bruitées = {((p > .5) == y[va]).mean():.4f}')
