# DPO — Alignement éthique de Qwen2.5-1.5B sur ETHICS

Volet DPO du projet M1 ADL *Ethical Alignment of Small Language Models*. Le code, les configs, les résultats et les figures de la **meilleure configuration** (run « max ») vivent à la racine. Les configurations intermédiaires et les artefacts produits pendant l'exploration sont archivés dans [`versions_precedentes/`](versions_precedentes/).

## Résultat principal

| Modèle                              | macro ETHICS | commonsense | deontology | justice | virtue |
|-------------------------------------|-------------:|------------:|-----------:|--------:|-------:|
| Baseline Qwen2.5-1.5B-Instruct      |        0.540 |       0.500 |      0.520 |   0.510 |  0.630 |
| **+ DPO (ETHICS-train, run "max")** |    **0.710** |   **0.630** |  **0.760** | **0.710** | **0.740** |
| **Δ**                               |     **+17 pp** | +13 pp     | +24 pp     | +20 pp  | +11 pp |

Évalué sur ETHICS test, 100 exemples par sous-ensemble équilibrés 50/50, scoring log-likelihood Yes/No. Détail méthodologique : [`report/REPORT.md`](report/REPORT.md). Comparaison des six configurations explorées : [`notebooks/results_analysis.ipynb`](notebooks/results_analysis.ipynb).

## Structure du dépôt

```
.
├── README.md                          ← vous êtes ici
├── CLAUDE.md                          briefing projet (cadrage M1 ADL)
├── Projet_ADL.pdf                     sujet original
├── Compte_rendu_ADL.docx              rapport groupe (RAG + RLHF + DPO)
├── report/REPORT.md                   rapport ACL draft, volet DPO uniquement
├── configs/
│   └── dpo_ethics_max.yaml            config de la meilleure run (LoRA r=16, β=0.05, 4000 paires)
├── src/
│   ├── data.py                        loaders PKU-SafeRLHF + ETHICS-train préférences
│   ├── model.py                       chargement Qwen + QLoRA + adaptateur
│   ├── train.py                       boucle DPO (TRL)
│   ├── eval.py                        scoring log-prob Yes/No sur ETHICS
│   └── utils.py                       seed, config YAML, truststore
├── scripts/
│   ├── eval_baseline.py               évalue Qwen2.5-1.5B sans DPO
│   ├── train_dpo.py                   entraînement (lit configs/dpo_ethics_max.yaml)
│   ├── eval_dpo.py                    évalue un adaptateur DPO sur ETHICS test
│   ├── qualitative.py                 génère baseline vs DPO sur ETHICS échecs + safety probes
│   ├── run_ablations.py               grille β/LR/data (non utilisée pour la run max)
│   └── update_docx.py                 injecte le contenu DPO dans Compte_rendu_ADL.docx
├── notebooks/
│   ├── results_analysis.ipynb         tableau + figures des 6 runs, matrices de confusion, qualitatif
│   └── report_figures.ipynb           génère les 3 figures embarquées dans le docx
├── results/
│   ├── baseline_eval_rtx.json         baseline 1.5B sans DPO
│   ├── dpo_eval_ethics_max.json       éval ETHICS de la run max
│   ├── qualitative_max.json           générations baseline vs DPO max
│   └── fig_*.png                      figures (5 PNG : 2 du notebook + 3 du docx)
├── logs/
│   ├── 01_baseline_rtx.log            log du eval_baseline
│   ├── 02_train_ethics_max.log        log du train_dpo (3h sur RTX 4070 Laptop)
│   ├── 03_eval_dpo_ethics_max.log     log du eval_dpo
│   └── 04_qualitative_max.log         log du qualitative.py
├── requirements.txt                   deps GPU (bitsandbytes, cu121)
├── requirements-cpu.txt               deps CPU fallback (smoke test sans GPU)
└── versions_precedentes/              configs / résultats / logs / notebook / docx originaux
    │                                  des 5 runs intermédiaires (PKU full, PKU step-200,
    │                                  ETHICS v1, ETHICS v2, run v3 avortée, et la version
    │                                  du compte-rendu avant injection DPO)
    └── README.md                      détail run par run
```

## Reproduire la run principale

### Pré-requis matériel
- GPU NVIDIA avec ≥ 8 Go VRAM et architecture Ampere ou Ada (RTX 30/40+) pour bfloat16 natif. La run originale a tourné sur **RTX 4070 Laptop (8 Go)** en ~3 h. Sur T4 Colab (16 Go) il faut basculer en `fp16` (voir `versions_precedentes/configs/dpo_t4.yaml` pour le précédent qui fonctionnait).
- Python 3.11.

### Installation
```powershell
.\.venv\Scripts\Activate.ps1     # ou source .venv/bin/activate sous Linux/Colab
pip install -r requirements.txt
```

Si `bitsandbytes` refuse de s'importer sous Windows, désactiver `model.load_in_4bit` dans la config (le modèle prendra ~3 Go au lieu de 1 Go en VRAM mais ça tient toujours sur la 4070).

### Pipeline complète

```powershell
# 1. Baseline (Qwen2.5-1.5B sans DPO) — ~3 min
python -m scripts.eval_baseline `
    --config configs/dpo_ethics_max.yaml `
    --out results/baseline_eval_rtx.json

# 2. Training DPO sur ETHICS-train (4000 paires synthétiques) — ~3 h
python -m scripts.train_dpo --config configs/dpo_ethics_max.yaml

# 3. Évaluation de l'adaptateur DPO — ~3 min
python -m scripts.eval_dpo `
    --config configs/dpo_ethics_max.yaml `
    --adapter checkpoints/dpo_ethics_max/final `
    --out results/dpo_eval_ethics_max.json

# 4. Analyse qualitative (générations baseline vs DPO sur ETHICS échecs + safety probes) — ~5 min
python -m scripts.qualitative `
    --config configs/dpo_ethics_max.yaml `
    --adapter checkpoints/dpo_ethics_max/final `
    --baseline-eval results/baseline_eval_rtx.json `
    --out results/qualitative_max.json
```

Les notebooks [`results_analysis.ipynb`](notebooks/results_analysis.ipynb) et [`report_figures.ipynb`](notebooks/report_figures.ipynb) consomment les JSON ci-dessus et régénèrent les figures.

## Méthode en 5 lignes

1. **Base** : Qwen2.5-1.5B-Instruct quantifié en NF4 (QLoRA) + LoRA r=16 (q/k/v/o + MLP gate/up/down), ~12 M params trainables.
2. **Préférences** : 4000 paires synthétisées depuis ETHICS *train* (`chosen` = label gold, `rejected` = label opposé). ETHICS *test* n'est jamais vu.
3. **DPO** : β = 0.05, LR = 2e-5 (cosine, 10 % warmup), batch effectif 8 (per_device = 1 × grad_accum = 8), 500 pas optimizer, paged_adamw_8bit, bf16 natif sur Ada.
4. **Politique de référence** : le même modèle avec adaptateur LoRA désactivé (`ref_model=None` dans TRL DPOTrainer) — un seul jeu de poids résident en VRAM.
5. **Évaluation** : log-likelihood des tokens "Yes" / "No" sous le chat template Qwen, argmax. 100 exemples par sous-ensemble (commonsense, deontology, justice, virtue), équilibrés 50/50.

## Mettre à jour le compte-rendu .docx

```powershell
python scripts/update_docx.py
```

Restaure `Compte_rendu_ADL.docx` depuis la version originale (sauvegardée dans `versions_precedentes/Compte_rendu_ADL_original.docx`), réinjecte les 6 sections DPO + la ligne du tableau + la nouvelle sous-section §4.1.b + les 3 figures. Idempotent.

## Pour reproduire une run précédente

Les configs et résultats des 5 runs intermédiaires sont dans [`versions_precedentes/`](versions_precedentes/). Voir [`versions_precedentes/README.md`](versions_precedentes/README.md) pour la commande exacte à lancer (les chemins de config et d'output sont différents).

## Références

- Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023
- Hendrycks et al., *Aligning AI With Shared Human Values*, ICLR 2021
- Ji et al., *PKU-SafeRLHF*, NeurIPS Datasets & Benchmarks 2024
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
