# Volet RLHF — section du compte-rendu (miroir versionné)

Ce fichier reproduit, pour le suivi Git, le contenu inséré dans le compte-rendu
partagé `Compte_rendu_ADL.docx` (sections 2.3, 2.5, 3.2, 4.1, 4.3). La version
de référence reste le `.docx` co-édité sur le Drive ; ce miroir sert à
versionner le texte et les chiffres.

## 2.3 — Méthode

Pipeline en trois temps (Ouyang et al., 2022), sans l'étape SFT (le modèle de
base `Qwen2.5-1.5B-Instruct` est déjà instruction-tuné). On garde (2) un modèle
de récompense et (3) l'optimisation de la politique, tous deux sur le même
Qwen2.5-1.5B.

**Modèle de récompense** : tête de classification scalaire (`num_labels=1`),
QLoRA 4 bits (NF4, bf16, double quant), adaptateurs LoRA `r=16, α=32,
dropout=0.05` sur `q,k,v,o` (~9 M params entraînables). `RewardTrainer` (TRL),
perte de Bradley-Terry `−log σ(r_choisi − r_rejeté)`. 1 époque, batch effectif
16 (4×4), lr `5e-5`, `max_length=512`, seed 42.

**Optimisation — RLOO plutôt que PPO** (la consigne autorise une alternative
légère à PPO) : (1) TRL 1.5 a retiré `PPOTrainer` ; (2) RLOO supprime value head
+ GAE — pour chaque prompt, K complétions, baseline = moyenne des K−1 autres →
moitié moins de mémoire (décisif sur T4). Pénalité KL `β=0.04` vers une politique
de référence gelée (garde-fou anti reward-hacking).
Config RLOO : `K=4`, 500 prompts, 1 époque, lr `1e-6`, batch 4 × accum 2,
`β=0.04`, T=1.0, complétions ≤ 64 tokens (prompts ≤ 256).

## 2.5 — Données

`Anthropic/HH-RLHF` (préférences humaines helpful & harmless). 3 000 paires
train / 300 eval, chat template Qwen, troncature 512. Les mêmes prompts (sans
réponse) → 500 invites RLOO. **Aucune** donnée synthétique, **aucun** LLM
externe. Pas de fuite vers ETHICS (jeux indépendants, ETHICS en test seulement).

## 3.2 — Setup

GPU T4 16 Go (Colab/Kaggle gratuit), QLoRA 4 bits + LoRA, AdamW, seed 42.
Temps : ~1 h reward model (174 pas), ~2 h RLOO (249 pas), ~25 min éval.
Checkpointé. **Éval ETHICS du volet RLHF sur 50 ex/sous-ensemble** (budget T4),
avec son propre baseline mesuré dans les mêmes conditions.

## 4.1 — Résultats quantitatifs

| Méthode | CS | DE | JU | VI | UT | Moy. |
|---|---|---|---|---|---|---|
| RLHF (baseline interne) | 62.0 | 50.0 | 62.0 | 80.0 | 56.0 | 62.0 |
| RLHF (politique alignée) | 62.0 | 50.0 | 62.0 | 80.0 | 58.0 | 62.4 |

> ⚠️ Harnais distinct des lignes Baseline/RAG du compte-rendu (50 vs 100 ex,
> baseline interne 62.0 et non 59.0). Seul l'écart RLHF − baseline interne est
> interprétable : **+0,4 pt** moyen, porté par le seul utilitarianism (56 → 58).

- **Reward model** (`figures/reward_training.png`) : perte de validation ~0,688
  (< ln 2 ≈ 0,693), exactitude des préférences ~0,557 (max 0,567), marge ~+0,05.
  Signal réel mais **faible** (~56 %) → plafonne ce que le RL peut exploiter.
- **RLOO** (`figures/rloo_training.png`) : récompense moyenne ≈ 5,6 (stable, ne
  progresse pas), KL ~1e-3/token, entropie ~1,2 → la politique bouge très peu.
- **ETHICS** (`figures/ethics_comparison.png`) : 62,0 % → 62,4 %, seul l'UT gagne.

**Lecture** : résultat quasi nul mais cohérent — (i) reward model faible,
(ii) KL + lr bas → politique proche du départ, (iii) décalage de distribution
HH-RLHF (dialogue) vs ETHICS (classification morale). Même leçon que le RAG : sur
un petit modèle déjà aligné, une couche d'alignement de plus n'apporte qu'un gain
marginal ; la valeur du RLHF est la maîtrise du pipeline démontrée.

## 4.3 — Qualitatif

Décisions base vs RLHF quasi identiques (scoring log-prob) sur CS/DE/JU/VI —
confirme le faible déplacement KL. Différences concentrées sur l'utilitarianism
(quelques comparaisons de bien-être corrigées). Illustre la prudence de RLOO sous
forte régularisation KL.

## Reproduire les figures

```bash
python report/fill_rlhf_section.py   # remplit le .docx + ré-insère les figures
```

Sources des chiffres : `outputs/reward_model/checkpoint-174/trainer_state.json`,
`outputs/rloo_policy/checkpoint-249/trainer_state.json`,
`outputs/eval/results.json`.
