# Mémo : ce que le projet Posos peut apporter à Kili

> Écrit le 2026-10-23 depuis `~/Documents/Hugging-Face/posos_camembert` (challenge ENS n°17,
> classification de questions médicales, 51 classes ; 0,7583 en 2019 → 0,8124 aujourd'hui).
> Ce mémo ne remplace pas `NOTE_PASSATION.md` : il rassemble ce qui a été **mesuré** sur Posos
> et qui sert directement un challenge data-centric. Le code cité est dans `posos_camembert/`.

## 1. Trouver les étiquettes fausses : prédictions out-of-fold
C'est la technique centrale pour Kili (environ 25 % d'étiquettes fausses d'après `match_imdb.csv`).

- Entraîner un modèle fort en **validation croisée 5 plis** : chaque exemple est prédit par un
  modèle qui ne l'a pas vu. Code prêt : `train.py --folds 5 --fold i` sauve `val_probs.npy` et
  `val_ids.npy` ; `ensemble.py oof "motif*"` réassemble les 5 plis et évalue.
- **Étiquettes suspectes = désaccords à forte confiance.** Sur Posos (étiquettes humaines, non
  bruitées volontairement), 232 erreurs sur 8 028 avaient une confiance > 80 %, et à la lecture
  le modèle avait souvent raison contre l'étiquette. Sur Kili, le bruit est volontaire, donc ce
  signal devrait être bien plus fort.
- Un **ensemble** de modèles différents juge mieux qu'un seul (Posos : +1,4 pt d'accuracy en
  passant de 1 à 4 modèles).
- **Contrôle gratuit sur Kili** : les ~2 900 critiques retrouvées dans IMDB (similarité ≥ 0,99)
  ont une étiquette d'origine connue. Elles permettent de mesurer la précision du détecteur
  d'erreurs (combien de suspects sont vraiment faux) avant de l'appliquer aux 7 000 autres.

## 2. Le meilleur « professeur » : un LLM de 7B affiné en classifieur (QLoRA, Colab gratuit)
Résultat le plus fort de Posos : **Qwen2.5-7B en 4 bits + LoRA, sur une T4 Colab gratuite**,
bat tous les CamemBERT (0,812 contre 0,804 au mieux sur le même pli), et l'ensemble avec lui
gagne **+1 à +1,7 pt**. Pour Kili, c'est le candidat naturel pour ré-étiqueter les données.

- Notebook prêt à adapter : `posos_camembert/colab/posos_qwen7b_lora.ipynb` (QLoRA nf4,
  r=16, toutes les projections, lr 1e-4, 2 epochs, environ 40 min pour 6 400 textes courts).
  Pour Kili : textes en anglais et plus longs, donc augmenter `MAX_LEN` (vérifier la mémoire)
  et prévoir plus de temps.
- **⚠️ Bug à connaître (il nous a coûté une heure)** : avec **transformers ≥ 4.46**,
  `group_by_length=True` trie aussi les données en **prédiction**. Les probabilités ressortent
  dans le désordre, alors que l'accuracy affichée pendant l'entraînement reste juste
  (prédictions et labels sont triés ensemble). Symptôme : accuracy « 0,08 » en recalculant à
  partir des fichiers. Correctif avant `trainer.predict` :
  `trainer.args.group_by_length = False`.
  En local (transformers 4.45), le problème n'existe pas : c'est la version installée par Colab
  qui l'a déclenché.

## 3. Un LLM comme arbitre des cas ambigus
Mesuré sur Posos : quand l'ensemble hésite entre deux classes, Qwen 14B local (Ollama),
nourri d'une description de chaque classe et de 3 exemples, **a corrigé 54 réponses et en a
cassé 16** sur 720 cas (+0,47 pt). Code : `posos_camembert/agent/arbitre.py`.
- Pour Kili : un second avis pour les textes où le professeur hésite ou contredit l'étiquette.
  On peut aussi s'en servir pour repérer le **hors-domaine** (« est-ce une critique de
  film ? »).
- Limite constatée : une fois le modèle 7B ajouté à l'ensemble, l'arbitre n'apportait presque
  plus rien (+2 réponses nettes sur 144). Il sert surtout quand le modèle de base est faible.

## 4. Méthode : ce qui a compté le plus
1. **Évaluation locale fiable avant tout.** Sur Posos, une validation de 1 205 exemples avait un
   bruit d'environ ±1 pt et masquait les vrais effets ; la validation croisée sur tout le jeu l'a
   réglé. Pour Kili, le test est caché et le train bruité : construire un jeu d'évaluation
   **propre** (par exemple IMDB) avec le **script fastText imposé**.
2. **Une seule modification à la fois**, toujours mesurée. Sur Posos, des idées séduisantes ont
   échoué à la mesure (lissage des étiquettes : −0,4 pt sur l'ensemble), et une idée donnée
   perdante a gagné (l'arbitre LLM).
3. **Analyser les erreurs tôt.** Sur Posos, elles portaient sur la nuance de la question, pas
   sur le vocabulaire : ça a évité de perdre du temps sur le prétraitement.
4. **Hyperparamètres de base d'abord** : le learning rate valait +3 pts sur Posos, plus que tout
   le reste. Ici fastText est imposé, mais c'est valable pour le modèle professeur.
5. **Registre des soumissions** (`posos_camembert/subs.py`) : fichier + réglages + score
   plateforme. Il faut l'adapter, puisqu'ici une soumission est un jeu de données.

## 5. Matériel et pièges pratiques (GTX 1080 Ti, Pascal)
- **Pas de fp16** : environ 45 % plus lent qu'en fp32 (pas de Tensor Cores). Le fp16 est utile
  sur la T4 de Colab.
- `group_by_length=True` en entraînement (moins de padding), `save_total_limit=1`
  (sinon environ 1,3 Go par epoch), `set_seed()` **avant** de créer le modèle (la tête de
  classification est initialisée au hasard).
- Ollama (Qwen 14B) prend environ 10 Go : **pas en même temps qu'un entraînement** sur le
  même GPU. Libérer avec `ollama stop <modèle>`.
- Un entraînement s'est **bloqué 2 h sans erreur** (log figé, GPU à 3 %) : surveiller la date
  de dernière modification des logs pendant les longs runs.
- Mise en veille automatique en fin de calcul : `systemctl suspend` fonctionne sans sudo.

## 6. Point éthique à trancher sur Kili
La moitié des critiques retrouvées dans IMDB viennent de son **split test**. Si le test caché de
Kili vient d'IMDB, y ajouter des critiques du test IMDB avec leurs étiquettes reviendrait à
entraîner sur le test. Corriger les étiquettes des critiques **déjà fournies** grâce à IMDB, ou
compléter avec le **train** IMDB, est plus défendable. À vérifier dans le règlement avant de
soumettre.
