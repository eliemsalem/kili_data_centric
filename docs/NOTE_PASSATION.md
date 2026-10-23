# Note de passation — Kili « Data Centric Movie Reviews Sentiment Classification »

> À lire en premier pour reprendre le projet. Créé le 2026-10-23.
> Projet frère : `~/Documents/Hugging-Face/posos_camembert` (même méthode, voir « Leçons »).

## Le challenge (ENS Challenge Data n°61, Kili Technology)
https://challengedata.ens.fr/participants/challenges/61/

- **Tâche** : sentiment (positif 1 / négatif 0) de critiques de films, métrique = accuracy.
- **Particularité « data-centric »** : le MODÈLE EST IMPOSÉ (fastText, script fourni dans les
  fichiers supplémentaires). On soumet un **jeu d'entraînement** (même format que l'entrée,
  colonnes `review`, `category`), **20 000 lignes max** (au-delà : score 0). Kili entraîne
  fastText dessus (~60 s) et l'évalue sur un test caché de 10 000 critiques.
- **Données fournies** : 10 000 critiques **volontairement bruitées** : erreurs d'étiquette,
  textes venant d'autres jeux de données (hors domaine), augmentations…
  10 exemples du test sont révélés (pour voir à quoi ressemble la distribution).
- **Benchmark** : soumettre le jeu fourni tel quel.
- « You will be allowed to give anything to the model » → ajouter des données externes
  (sourcing) semble autorisé, dans la limite de 20 000 lignes. **À confirmer dans le règlement.**

## État (mis à jour le 2026-10-23, 2e session)
- [x] Données dans `data/` : `full_train.csv` (= `x_baseline.csv` + `y_baseline.csv`, 10 000 lignes,
      5 087 positifs / 4 913 négatifs), `train.py.zip` extrait dans `data/kili_script/train.py`.
      Les 10 exemples de test révélés ne sont PAS dans `data/` (à récupérer sur le site si besoin).
- [x] Script Kili lu : fastText `train_supervised(lr=0.1, epoch=20, wordNgrams=2)`, texte brut
      (aucun prétraitement), écrit via `to_csv(quoting=QUOTE_NONE, escapechar="\\")`.
      → 14 critiques contiennent un saut de ligne, qui casse le format fastText : les retirer
      dans nos soumissions.
- [x] Env conda **`kili`** (clone de `posos` + `fasttext-wheel` + `cleanlab`).
      Attention : numpy 2.4 ; si `model.predict` plante (`copy=False`), passer à numpy<2.
- [x] IMDB téléchargé (HF `stanfordnlp/imdb`) : `data/imdb/{train,test,unsupervised}.parquet`.
- [x] `src/match_imdb.py` → `data/match_imdb.csv` : critique IMDB la plus proche de chaque ligne
      (colonnes `sim`, `imdb_idx`, `split`, `imdb_label`, `category`). Résultats :
  - **97 % des lignes viennent d'IMDB** (sim > 0,5) : 4 879 du split train, 4 799 du split test ;
    ~3 % hors domaine (traductions en hindi, critiques d'albums…) ;
  - **accord étiquette Kili / IMDB = 75 %** → ~25 % d'étiquettes inversées ;
  - autres bruits : fautes de frappe artificielles (`nic3`, `m(vie`), tokenisation ` ' ` espacée,
    `<br />` ; 4 doublons exacts.
- [x] **Évaluation locale** (`src/common.py`) : `score(df)` = leur `train_model`, moyenne de 3 runs,
      sur `data/eval_imdb.csv` (10 000 critiques IMDB propres, absentes du train Kili). Bruit ±0,0005.
      Python 3.11 refuse `quotechar=""` : `common.py` remplace `process_csv` par une copie avec
      `quotechar=None` (même sortie). Les sauts de ligne sont retirés avant l'entraînement.
- [x] `src/bench.py` : **benchmark 0,7816** ; borne haute avec les étiquettes IMDB (mesure
      seulement) 0,8394.
- [x] `src/relabel_oof.py` (TF-IDF mots + caractères, régression logistique, OOF 5 plis →
      `data/oof_lr.npy`) ; le professeur est d'accord avec IMDB à 78,9 % :

      | seuil p(autre) | suspects | précision | rappel | inverser | supprimer |
      |---|---|---|---|---|---|
      | 0,5 | 3 542 | 0,555 | 0,780 | 0,8129 | 0,7969 |
      | 0,6 | 2 153 | 0,706 | 0,604 | **0,8130** | 0,8050 |
      | 0,7 | 1 135 | 0,843 | 0,383 | 0,8115 | 0,8031 |
      | 0,8 | 442 | 0,940 | 0,169 | 0,8049 | 0,7968 |

      → **inverser > supprimer** ; +3,1 pts. Écart restant avec la borne haute : 2,6 pts.
- [x] `src/teacher_oof.py` roberta-base OOF 5 plis sur la 1080 Ti (~15 min/pli, logs dans
      `logs/`, sorties `data/oof/roberta-base_f*.npz` → `data/oof_roberta-base.npy`).
      Accord avec IMDB **0,896** (régression logistique : 0,789). `src/relabel_teacher.py` :

      | professeur | seuil | suspects | précision | rappel | inverser |
      |---|---|---|---|---|---|
      | roberta | 0,5 | 3 102 | 0,736 | 0,909 | 0,8346 |
      | roberta | **0,6** | 2 685 | 0,802 | 0,867 | **0,8351** |
      | roberta | 0,7 | 2 073 | 0,893 | 0,753 | 0,8349 |
      | roberta + lr | 0,5 | 3 062 | 0,748 | 0,908 | 0,8344 |

      → **0,8351, à 0,4 pt de la borne haute (0,8394)**. La correction des étiquettes est
      presque épuisée ; le seuil compte peu entre 0,5 et 0,7. Les prochains gains viendront
      d'ailleurs (hors domaine, doublons, compléter jusqu'à 20 000 lignes, bruit de frappe).
- [x] `src/clean_ood.py` (base = inversion roberta t=0,6, 0,8352) : retirer le hors domaine ou
      les doublons **n'aide pas** (doublons 0,8348 ; 21 non-ASCII 0,8350 ; 322 sim < 0,5, borne
      haute, 0,8330). fastText supporte bien ce bruit ; moins de lignes = moins d'apprentissage.
- [x] `src/augment.py` : **doubler le jeu jusqu'à 20 000 lignes = +1,3 à +1,5 pt** (équivaut à
      plus d'epochs, le script Kili étant fixé à 20) :
      normalisé seul (`' ` recollé, ponctuation) 0,8380 ; x2 identique 0,8487 ; x2 normalisé
      0,8477 ; **base + copie normalisée 0,8498** (meilleur, la diversité aide).
      ⚠ Le gain de la normalisation suppose un test à la typographie d'IMDB brut, comme notre
      évaluation : **vérifier sur les 10 exemples de test révélés** (pas encore téléchargés).
- [ ] Prêt : `colab/kili_qwen7b_oof.ipynb` (Qwen2.5-7B QLoRA, 1 epoch, 128 + 384 tokens, mêmes plis,
      reprise via Drive). Durée par pli inconnue (estimée 2 à 3 h sur T4) : lancer le pli 0 seul d'abord.
- [ ] Prochaine étape : un professeur plus fort (transformer affiné en OOF 5 plis, ou Qwen 7B
      QLoRA, cf. mémo Posos) pour se rapprocher de 0,839.
- [x] Recherche d'un ancien code de l'utilisateur sur `/media/elie/265C-8015` (vers le
      2022-05-01) : **rien trouvé**. Aucun fichier daté du 2022-04-28 au 2022-05-03, aucun nom
      contenant « kili » ; seulement des tutoriels de sentiment et des copies d'IMDB (`aclImdb`)
      datant de 2018-2021. S'il existe, il est ailleurs (autre disque, Colab, Drive ?).

## Mémo Posos
Lire aussi `docs/MEMO_DEPUIS_POSOS.md` : out-of-fold, professeur Qwen 7B QLoRA (bug `group_by_length`),
arbitre LLM, pièges de la 1080 Ti.
Correction de son point 1 : le jeu de contrôle n'est pas limité aux ~2 900 critiques avec
sim ≥ 0,99 (textes intacts). Toutes les tranches de similarité > 0,5 ont le même accord
d'étiquette (74,6 % pour ≥ 0,99, 74,6 % pour [0,7 ; 0,99[, 75,8 % pour [0,5 ; 0,7[). Avec de
mauvais appariements, cet accord tomberait vers 50 % : les correspondances sont donc fiables, et
**~9 680 lignes** ont une étiquette IMDB connue.

## ⚠ Question ouverte : règlement (à trancher AVANT d'utiliser IMDB)
Le test caché est presque sûrement composé de critiques IMDB lui aussi. Deux raccourcis seraient
très efficaces, mais ce serait peut-être une fuite de données :
- ré-étiqueter le train avec les étiquettes IMDB officielles (`imdb_label`) ;
- compléter jusqu'à 20 000 lignes avec IMDB public, qui peut contenir des critiques du test caché.
Tant que ce n'est pas confirmé dans le règlement, s'en tenir aux méthodes « légitimes »
(prédictions out-of-fold, modèle professeur). `imdb_label` peut quand même servir à **mesurer** la
qualité d'un nettoyage (précision / rappel de la détection des étiquettes inversées).

## Évaluation locale : attention
La moitié du train vient du split test d'IMDB : le test IMDB n'est donc **pas** un proxy propre tel
quel. Proxy proposé : les critiques IMDB (train + test, étiquettes officielles) qui ne correspondent
à **aucune** ligne du train Kili (exclure tous les `imdb_idx` de `match_imdb.csv`). Échantillonner
10 000 critiques ; garder la même graine ; si possible, leur appliquer le même bruit de frappe que
le train, puisqu'on ignore si le test caché en contient.

## Plan
0. **Lire le script fourni** : prétraitement fait par Kili, hyperparamètres fastText, format exact.
   Notre évaluation locale doit utiliser exactement le même entraînement.
1. **Évaluation locale fiable** : le test est caché et nos étiquettes sont bruitées, donc un
   simple découpage du train ne suffit pas. Pistes : comparer les 10 exemples de test à IMDB ;
   si c'est IMDB, utiliser le jeu de test IMDB public (25 000 critiques propres) comme proxy.
2. **Diagnostic du bruit** (mesurer avant d'agir) :
   - textes hors domaine (pas des critiques de films, autre langue…) ;
   - doublons et quasi-doublons, augmentations ;
   - **erreurs d'étiquette** : prédictions out-of-fold d'un modèle fort (validation croisée
     5 plis), et désaccords à forte confiance (méthode déjà utilisée sur Posos ; cf. aussi
     « confident learning » / bibliothèque cleanlab).
3. **Interventions, une à la fois, chacune mesurée en local** :
   a. supprimer le hors-domaine et les doublons ;
   b. corriger ou supprimer les étiquettes suspectes (comparer les deux) ;
   c. ré-étiqueter avec un modèle « professeur » fort (transformer anglais de sentiment) ;
   d. compléter jusqu'à 20 000 lignes avec des critiques externes (IMDB, etc.), si autorisé ;
   e. prétraitement du texte : seulement s'il rapproche le train du test (fastText voit le
      test brut, on ne contrôle pas son prétraitement).
4. **Soumissions** : vérifier le nombre autorisé par jour ; ne soumettre qu'un jeu qui bat le
   précédent en local ; archiver chaque soumission (fichier, nb de lignes, score local, score
   plateforme).

## Leçons reprises de Posos
- **Mesurer chaque changement séparément** sur une évaluation locale fiable, et se méfier du
  bruit (Posos : ±1 pt sur 1 205 questions, d'où la validation croisée sur tout le train).
- **Validation croisée dès le départ** pour tout modèle qui juge les données.
- **Analyser les erreurs tôt** au lieu d'empiler des idées au hasard.
- **Registre des soumissions** avec réglages + score plateforme (Posos : `subs.py`). Ici, une
  soumission est un jeu de données, pas des prédictions : à adapter.
- **GTX 1080 Ti (Pascal)** : pas de fp16 (~45 % plus lent), `group_by_length`, limiter les
  checkpoints, fixer le seed AVANT de créer le modèle. Env conda dédié (comme `posos`), torch 2.4.1+cu121.
- Ne jamais publier les données du challenge ni les soumissions (`.gitignore`).
