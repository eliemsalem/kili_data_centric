# Handoff note: Kili "Data Centric Movie Reviews Sentiment Classification"

> Read this first to pick the project up again. Created on 2026-10-23.
> Sibling project: `~/Documents/Hugging-Face/posos_camembert` (same method, see "Lessons").

## The challenge (ENS Challenge Data #61, Kili Technology)
https://challengedata.ens.fr/participants/challenges/61/

- **Task**: sentiment (positive 1 / negative 0) of movie reviews; metric = accuracy.
- **The "data-centric" twist**: THE MODEL IS FIXED (fastText, script provided in the
  supplementary files). You submit a **training set** (same format as the input,
  columns `review`, `category`), **20,000 rows at most** (above that: score 0). Kili trains
  fastText on it (~60 s) and evaluates it on a hidden test set of 10,000 reviews.
- **Provided data**: 10,000 **deliberately noisy** reviews: label errors, texts taken from
  other datasets (off-topic), augmentations…
  10 test examples are revealed (to show what the distribution looks like).
- **Benchmark**: submit the provided set as is.
- "You will be allowed to give anything to the model" → adding external data (sourcing)
  seems allowed, within the 20,000-row limit. **To be confirmed in the rules.**

## Project paused (2026-10-23)
- **Best set: `submissions/sub01_roberta_x2norm.csv`, local score 0.8494** (benchmark 0.7816).
- **Submission impossible**: the platform returns `invalid_grant: account not found`, coming from
  Kili's external API, which appears to be out of service. No platform score.
- Qwen 7B fold 0 (Colab) was not used: possible gain ≤ 0.4 points (see roberta).
- Public code: https://github.com/eliemsalem/kili_data_centric (without data or submissions).
- To resume: reactivate the `kili` conda env (or `pip install -r requirements.txt`), and
  first check whether the platform scores again (submit `data/full_train.csv` as is).

## Status (updated 2026-10-23, 2nd session)
- [x] Data in `data/`: `full_train.csv` (= `x_baseline.csv` + `y_baseline.csv`, 10,000 rows,
      5,087 positive / 4,913 negative), `train.py.zip` extracted to `data/kili_script/train.py`.
      The 10 revealed test examples are NOT in `data/` (get them from the site if needed).
- [x] Kili's script read: fastText `train_supervised(lr=0.1, epoch=20, wordNgrams=2)`, raw text
      (no preprocessing), written with `to_csv(quoting=QUOTE_NONE, escapechar="\\")`.
      → 14 reviews contain a line break, which breaks the fastText format: remove them
      from our submissions.
- [x] Conda env **`kili`** (clone of `posos` + `fasttext-wheel` + `cleanlab`).
      Warning: numpy 2.4; if `model.predict` crashes (`copy=False`), switch to numpy<2.
- [x] IMDB downloaded (HF `stanfordnlp/imdb`): `data/imdb/{train,test,unsupervised}.parquet`.
- [x] `src/match_imdb.py` → `data/match_imdb.csv`: closest IMDB review for each row
      (columns `sim`, `imdb_idx`, `split`, `imdb_label`, `category`). Results:
  - **97% of the rows come from IMDB** (sim > 0.5): 4,879 from the train split, 4,799 from the
    test split; ~3% off-topic (Hindi translations, album reviews…);
  - **Kili / IMDB label agreement = 75%** → ~25% of labels flipped;
  - other noise: artificial typos (`nic3`, `m(vie`), spaced ` ' ` tokenisation,
    `<br />`; 4 exact duplicates.
- [x] **Local evaluation** (`src/common.py`): `score(df)` = their `train_model`, mean of 3 runs,
      on `data/eval_imdb.csv` (10,000 clean IMDB reviews, absent from the Kili training set).
      Noise ±0.0005. Python 3.11 rejects `quotechar=""`: `common.py` replaces `process_csv`
      with a copy using `quotechar=None` (same output). Line breaks are removed before training.
- [x] `src/bench.py`: **benchmark 0.7816**; upper bound with the IMDB labels (measurement
      only) 0.8394.
- [x] `src/relabel_oof.py` (word + character TF-IDF, logistic regression, 5-fold OOF →
      `data/oof_lr.npy`); the teacher agrees with IMDB 78.9% of the time:

      | threshold p(other) | suspects | precision | recall | flip | delete |
      |---|---|---|---|---|---|
      | 0.5 | 3,542 | 0.555 | 0.780 | 0.8129 | 0.7969 |
      | 0.6 | 2,153 | 0.706 | 0.604 | **0.8130** | 0.8050 |
      | 0.7 | 1,135 | 0.843 | 0.383 | 0.8115 | 0.8031 |
      | 0.8 | 442 | 0.940 | 0.169 | 0.8049 | 0.7968 |

      → **flip > delete**; +3.1 points. Remaining gap to the upper bound: 2.6 points.
- [x] `src/teacher_oof.py` roberta-base 5-fold OOF on the 1080 Ti (~15 min/fold, logs in
      `logs/`, outputs `data/oof/roberta-base_f*.npz` → `data/oof_roberta-base.npy`).
      IMDB agreement **0.896** (logistic regression: 0.789). `src/relabel_teacher.py`:

      | teacher | threshold | suspects | precision | recall | flip |
      |---|---|---|---|---|---|
      | roberta | 0.5 | 3,102 | 0.736 | 0.909 | 0.8346 |
      | roberta | **0.6** | 2,685 | 0.802 | 0.867 | **0.8351** |
      | roberta | 0.7 | 2,073 | 0.893 | 0.753 | 0.8349 |
      | roberta + lr | 0.5 | 3,062 | 0.748 | 0.908 | 0.8344 |

      → **0.8351, 0.4 points from the upper bound (0.8394)**. Label correction is almost
      exhausted; the threshold matters little between 0.5 and 0.7. The next gains will come
      from elsewhere (off-topic texts, duplicates, filling up to 20,000 rows, typo noise).
- [x] `src/clean_ood.py` (base = roberta flipping t=0.6, 0.8352): removing off-topic texts or
      duplicates **does not help** (duplicates 0.8348; 21 non-ASCII 0.8350; 322 sim < 0.5, upper
      bound, 0.8330). fastText handles this noise well; fewer rows = less learning.
- [x] `src/augment.py`: **doubling the set to 20,000 rows = +1.3 to +1.5 points** (equivalent to
      more epochs, since Kili's script is fixed at 20):
      normalised only (`' ` rejoined, punctuation) 0.8380; x2 identical 0.8487; x2 normalised
      0.8477; **base + normalised copy 0.8498** (best, diversity helps).
      ⚠ The normalisation gain assumes a test set with raw IMDB typography, like our
      evaluation: **check on the 10 revealed test examples** (not downloaded yet).
- [ ] Ready: `colab/kili_qwen7b_oof.ipynb` (Qwen2.5-7B QLoRA, 1 epoch, 128 + 384 tokens, same folds,
      resumes via Drive). Time per fold unknown (estimated 2 to 3 h on a T4): run fold 0 alone first.
- [ ] Next step: a stronger teacher (transformer fine-tuned 5-fold OOF, or Qwen 7B
      QLoRA, see the Posos memo) to get closer to 0.839.
- [x] Searched for an old piece of the user's code on `/media/elie/265C-8015` (around
      2022-05-01): **nothing found**. No file dated 2022-04-28 to 2022-05-03, no name
      containing "kili"; only sentiment tutorials and copies of IMDB (`aclImdb`)
      from 2018-2021. If it exists, it is elsewhere (another disk, Colab, Drive?).

## Posos memo
Also read `docs/MEMO_FROM_POSOS.md`: out-of-fold, Qwen 7B QLoRA teacher (`group_by_length` bug),
LLM arbiter, 1080 Ti pitfalls.
Correction to its point 1: the control set is not limited to the ~2,900 reviews with
sim ≥ 0.99 (intact texts). All similarity bands above 0.5 have the same label
agreement (74.6% for ≥ 0.99, 74.6% for [0.7, 0.99), 75.8% for [0.5, 0.7)). With
bad matches, this agreement would drop towards 50%: the matches are therefore reliable, and
**~9,680 rows** have a known IMDB label.

## ⚠ Open question: the rules (to settle BEFORE using IMDB)
The hidden test set is almost certainly made of IMDB reviews too. Two shortcuts would be
very effective, but they might be data leakage:
- relabelling the training set with the official IMDB labels (`imdb_label`);
- filling up to 20,000 rows with public IMDB, which may contain reviews from the hidden test.
Until this is confirmed in the rules, stick to "legitimate" methods
(out-of-fold predictions, teacher model). `imdb_label` can still be used to **measure** the
quality of a cleaning step (precision / recall of flipped-label detection).

## Local evaluation: caution
Half of the training set comes from the IMDB test split: the IMDB test set is therefore **not** a
clean proxy as is. Proposed proxy: the IMDB reviews (train + test, official labels) that match
**no** row of the Kili training set (exclude every `imdb_idx` in `match_imdb.csv`). Sample
10,000 reviews; keep the same seed; if possible, apply the same typo noise as the training set,
since we do not know whether the hidden test contains any.

## Plan
0. **Read the provided script**: preprocessing done by Kili, fastText hyperparameters, exact format.
   Our local evaluation must use exactly the same training.
1. **Reliable local evaluation**: the test set is hidden and our labels are noisy, so a
   simple split of the training set is not enough. Ideas: compare the 10 test examples with IMDB;
   if they are IMDB, use the public IMDB test set (25,000 clean reviews) as a proxy.
2. **Noise diagnosis** (measure before acting):
   - off-topic texts (not movie reviews, another language…);
   - duplicates and near-duplicates, augmentations;
   - **label errors**: out-of-fold predictions from a strong model (5-fold
     cross-validation), and high-confidence disagreements (method already used on Posos; see also
     "confident learning" / the cleanlab library).
3. **Interventions, one at a time, each measured locally**:
   a. remove off-topic texts and duplicates;
   b. correct or delete suspect labels (compare both);
   c. relabel with a strong "teacher" model (English sentiment transformer);
   d. fill up to 20,000 rows with external reviews (IMDB, etc.), if allowed;
   e. text preprocessing: only if it brings the training set closer to the test set (fastText sees
      the raw test, we do not control its preprocessing).
4. **Submissions**: check the number allowed per day; only submit a set that beats the
   previous one locally; log every submission (file, number of rows, local score, platform
   score).

## Lessons from Posos
- **Measure every change separately** on a reliable local evaluation, and watch out for
  noise (Posos: ±1 point on 1,205 questions, hence cross-validation on the whole training set).
- **Cross-validation from the start** for any model that judges the data.
- **Analyse errors early** instead of stacking random ideas.
- **Submission log** with settings + platform score (Posos: `subs.py`). Here, a
  submission is a dataset, not predictions: adapt it.
- **GTX 1080 Ti (Pascal)**: no fp16 (~45% slower), `group_by_length`, limit
  checkpoints, set the seed BEFORE creating the model. Dedicated conda env (like `posos`), torch 2.4.1+cu121.
- Never publish the challenge data or the submissions (`.gitignore`).
