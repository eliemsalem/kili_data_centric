# Data-centric sentiment classification (Kili × ENS Challenge Data #61)

In this challenge **the model is fixed and the training data is what you submit**. Kili's
organisers train a fastText classifier on your training set (at most 20,000 rows) and score it
on a hidden test set of 10,000 movie reviews. The 10,000 training reviews they provide are
deliberately corrupted: labels are flipped, typos are injected, and some texts are off-topic or
in another language.

**Result: local accuracy rises from 0.7816 to 0.8494 (+6.8 points)** with the organisers' own
training script untouched. Only the data changes.

## Data

The challenge data is **not in this repository**: it may not be redistributed. It can be
downloaded, after creating a free account, from the challenge page:
**[ENS Challenge Data #61, Kili Technology](https://challengedata.ens.fr/participants/challenges/61/)**.

| File on the challenge page | Put it in | Content |
|---|---|---|
| `full_train.csv` (also split as `x_baseline.csv` + `y_baseline.csv`) | `data/` | 10,000 noisy reviews and their labels |
| supplementary files (`train.py.zip`) | unzip to `data/kili_script/` | the organisers' fastText training script |

The public IMDB dataset used for evaluation is downloaded by `src/download_imdb.py`, from
[Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb).

## Approach

1. **Build an evaluation I can trust before changing anything.** The test set is hidden and the
   training labels are noisy, so a split of the training data would not measure much. I matched
   every training review to its source in the public IMDB dataset with TF-IDF nearest
   neighbours (`src/match_imdb.py`), which gave two findings:
   - 97% of the reviews come from IMDB, half from its train split and half from its test split;
   - **only 75% of the given labels agree with the original IMDB labels**, so about one in four
     is flipped.

   The evaluation set is 10,000 IMDB reviews that appear **nowhere** in the training data. Each
   score is the mean of 3 runs of the organisers' `train_model`; the spread between runs stays
   within ±0.001.
2. **Measure one change at a time**, and keep the changes that did not help.
3. **Use the IMDB labels to measure, never to train.** They give the precision and recall of the
   label-error detector. Copying them into the submission would amount to training on data
   that may overlap the hidden test set (see [Caveats](#caveats)).

## Results

All rows are fastText accuracy with the organisers' script on the clean evaluation set.

| Training set | Rows | Accuracy |
|---|---|---|
| Provided data as is (benchmark) | 10,000 | 0.7816 |
| Labels flipped by out-of-fold TF-IDF + logistic regression | 10,000 | 0.8130 |
| **Labels flipped by out-of-fold roberta-base** (5 folds, threshold 0.6) | 10,000 | **0.8352** |
| *Upper bound: original IMDB labels (measurement only)* | 10,000 | *0.8394* |
| Roberta-flipped, 322 texts not found in IMDB removed (off-topic, translations) | 9,678 | 0.8330 ✗ |
| Roberta-flipped, IMDB-style typography restored | 10,000 | 0.8380 |
| Roberta-flipped, duplicated | 20,000 | 0.8487 |
| **Roberta-flipped + a typography-normalised copy** (submitted) | 20,000 | **0.8498** |

What the numbers show:

- **Label noise was the main problem.** Each training review is judged by a model that never
  saw it: out-of-fold predictions, the core idea of *confident learning*. Roberta recovers the
  true label 89.6% of the time, against 78.9% for TF-IDF, and closes 93% of the gap between
  the benchmark and the upper bound.
  - At the 0.6 threshold it flags 2,685 reviews: 80% of them are truly mislabelled, and 87% of
    all mislabelled reviews are caught.
  - **Flipping suspect labels always beats deleting the reviews** (0.813 against 0.805 at best
    with TF-IDF): a corrected example still carries signal.
- **Removing off-topic text made things worse.** fastText tolerates it, and every row removed
  is lost training signal.
- **Filling the 20,000-row budget was the second-largest gain**, worth more than the
  remaining room in label correction. The script's number of epochs is fixed at 20, so showing
  every review twice trains the model longer. A second copy with a different typography beats
  an identical copy.

A Qwen2.5-7B QLoRA teacher is prepared in `colab/` (Colab T4). It was set aside: with roberta
already 0.4 points from the upper bound, a better teacher can add at most 0.4 points.

## Repository layout

```
src/
  common.py           local evaluation: organisers' train_model + clean IMDB evaluation set
  download_imdb.py    downloads IMDB (Hugging Face stanfordnlp/imdb) into data/imdb/
  match_imdb.py       matches each training review to its IMDB source (TF-IDF, cosine)
  bench.py            benchmark and upper bound
  relabel_oof.py      out-of-fold TF-IDF + logistic regression teacher; flip vs delete
  teacher_oof.py      out-of-fold roberta-base teacher, one fold per run (GPU)
  relabel_teacher.py  assembles the folds; detector precision and recall; fastText score
  clean_ood.py        effect of removing off-topic texts and duplicates
  augment.py          duplication and typography normalisation
  make_submission.py  builds the 20,000-row submission, re-reads and re-scores it, logs it
colab/                Qwen2.5-7B QLoRA out-of-fold teacher (resumes on Google Drive)
docs/                 working notes in French (lab notebook, lessons from a sibling project)
```

## Reproducing

First download the challenge data as described in [Data](#data).

```bash
pip install -r requirements.txt
# put full_train.csv in data/, and unzip the organisers' train.py.zip into data/kili_script/
cd src
python download_imdb.py
python match_imdb.py                                     # -> data/match_imdb.csv
python bench.py                                          # benchmark and upper bound
python relabel_oof.py                                    # TF-IDF teacher
for f in 0 1 2 3 4; do python teacher_oof.py --fold $f; done   # ~15 min per fold on a GTX 1080 Ti
python relabel_teacher.py roberta-base
python augment.py
python make_submission.py sub01_roberta_x2norm
```

The organisers' script writes CSV with `quotechar=""`, which Python 3.11 and later reject.
`common.py` patches in an equivalent writer (`quotechar=None`, same output) and leaves their
file unchanged.

## Caveats

- **No platform score.** On 2026-10-23 the challenge's external scoring API answered
  `invalid_grant: account not found`; it appears to be out of service. All scores above are local.
- **The typography gain assumes the hidden test looks like raw IMDB**, as the evaluation set
  does. Duplication and label correction do not depend on that assumption.
- **IMDB as a source.** Relabelling with the original IMDB labels, or adding IMDB reviews up to
  20,000 rows, would likely score higher. Both were left aside: the hidden test is very probably
  IMDB too, and the rules do not say whether this is allowed.
