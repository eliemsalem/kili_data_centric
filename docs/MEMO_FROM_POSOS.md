# Memo: what the Posos project can bring to Kili

> Written on 2026-10-23 from `~/Documents/Hugging-Face/posos_camembert` (ENS challenge #17,
> medical question classification, 51 classes; 0.7583 in 2019 → 0.8124 today).
> This memo does not replace `HANDOFF_NOTE.md`: it gathers what was **measured** on Posos
> and directly serves a data-centric challenge. The code cited is in `posos_camembert/`.

## 1. Finding wrong labels: out-of-fold predictions
This is the core technique for Kili (about 25% wrong labels according to `match_imdb.csv`).

- Train a strong model with **5-fold cross-validation**: each example is predicted by a
  model that has not seen it. Ready-made code: `train.py --folds 5 --fold i` saves `val_probs.npy` and
  `val_ids.npy`; `ensemble.py oof "pattern*"` reassembles the 5 folds and evaluates.
- **Suspect labels = high-confidence disagreements.** On Posos (human labels, not
  deliberately noisy), 232 errors out of 8,028 had a confidence > 80%, and on reading them
  the model was often right against the label. On Kili, the noise is deliberate, so this
  signal should be much stronger.
- An **ensemble** of different models judges better than a single one (Posos: +1.4 accuracy points
  going from 1 to 4 models).
- **Free check on Kili**: the ~2,900 reviews found in IMDB (similarity ≥ 0.99)
  have a known original label. They measure the precision of the error detector
  (how many suspects are truly wrong) before applying it to the other 7,000.

## 2. The best "teacher": a 7B LLM fine-tuned as a classifier (QLoRA, free Colab)
Strongest result on Posos: **Qwen2.5-7B in 4 bits + LoRA, on a free Colab T4**,
beats every CamemBERT (0.812 against 0.804 at best on the same fold), and the ensemble with it
gains **+1 to +1.7 points**. For Kili, it is the natural candidate to relabel the data.

- Notebook ready to adapt: `posos_camembert/colab/posos_qwen7b_lora.ipynb` (QLoRA nf4,
  r=16, all projections, lr 1e-4, 2 epochs, about 40 min for 6,400 short texts).
  For Kili: English and longer texts, so increase `MAX_LEN` (check memory)
  and plan for more time.
- **⚠️ Bug to know about (it cost us an hour)**: with **transformers ≥ 4.46**,
  `group_by_length=True` also sorts the data at **prediction** time. The probabilities come out
  in the wrong order, while the accuracy shown during training stays correct
  (predictions and labels are sorted together). Symptom: accuracy "0.08" when recomputing
  from the files. Fix before `trainer.predict`:
  `trainer.args.group_by_length = False`.
  Locally (transformers 4.45), the problem does not exist: the version installed by Colab
  triggered it.

## 3. An LLM as arbiter of ambiguous cases
Measured on Posos: when the ensemble hesitates between two classes, a local Qwen 14B (Ollama),
given a description of each class and 3 examples, **fixed 54 answers and
broke 16** out of 720 cases (+0.47 points). Code: `posos_camembert/agent/arbitre.py`.
- For Kili: a second opinion for texts where the teacher hesitates or contradicts the label.
  It can also be used to spot **off-topic** texts ("is this a movie
  review?").
- Observed limit: once the 7B model was added to the ensemble, the arbiter brought almost
  nothing (+2 net answers out of 144). It mostly helps when the base model is weak.

## 4. Method: what mattered most
1. **Reliable local evaluation first.** On Posos, a 1,205-example validation set had
   noise of about ±1 point and hid the real effects; cross-validation on the whole set
   fixed it. For Kili, the test is hidden and the training set noisy: build a **clean**
   evaluation set (for example IMDB) with the **fixed fastText script**.
2. **One change at a time**, always measured. On Posos, appealing ideas
   failed when measured (label smoothing: −0.4 points on the ensemble), and an idea expected
   to lose won (the LLM arbiter).
3. **Analyse errors early.** On Posos, they were about the nuance of the question, not
   the vocabulary: that avoided wasting time on preprocessing.
4. **Basic hyperparameters first**: the learning rate was worth +3 points on Posos, more than
   everything else. Here fastText is fixed, but it applies to the teacher model.
5. **Submission log** (`posos_camembert/subs.py`): file + settings + platform
   score. It needs adapting, since here a submission is a dataset.

## 5. Hardware and practical pitfalls (GTX 1080 Ti, Pascal)
- **No fp16**: about 45% slower than fp32 (no Tensor Cores). fp16 is useful
  on the Colab T4.
- `group_by_length=True` for training (less padding), `save_total_limit=1`
  (otherwise about 1.3 GB per epoch), `set_seed()` **before** creating the model (the
  classification head is randomly initialised).
- Ollama (Qwen 14B) takes about 10 GB: **not at the same time as a training run** on the
  same GPU. Free it with `ollama stop <model>`.
- A training run **froze for 2 h without an error** (log stuck, GPU at 3%): watch the
  last-modified time of the logs during long runs.
- Automatic suspend at the end of a computation: `systemctl suspend` works without sudo.

## 6. Ethical point to settle on Kili
Half of the reviews found in IMDB come from its **test split**. If Kili's hidden test
comes from IMDB, adding IMDB test reviews with their labels would amount to
training on the test. Correcting the labels of the reviews **already provided** with IMDB, or
filling up with the IMDB **train** split, is more defensible. Check the rules before
submitting.
