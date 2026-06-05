# versions_precedentes/

Archive des configurations DPO intermédiaires explorées avant la run finale (qui vit à la racine sous le nom `dpo_ethics_max`). Tout est fonctionnel — il suffit de lancer les scripts en pointant sur la config et l'output désirés.

## Les 5 runs intermédiaires

| ID interne       | Dataset           | Paires | Steps | LR     | LoRA       | β    | macro     | Δ baseline | Verdict                              |
|------------------|-------------------|-------:|------:|-------:|------------|-----:|----------:|-----------:|--------------------------------------|
| PKU full (T4)    | PKU-SafeRLHF      | 10 796 | 1 350 | 5e-6   | r=16+MLP   | 0.1  | 0.525     | −1.5 pp    | Transfer négatif ; PKU pousse au hedging |
| PKU step-200     | PKU-SafeRLHF      | 10 796 |   200 | 5e-6   | r=16+MLP   | 0.1  | 0.550     | +1.0 pp    | Early-stopping accidentel = mieux que full |
| ETHICS v1        | ETHICS-train      |  1 000 |   125 | 5e-6   | r=8 q,k,v,o| 0.1  | 0.5425    | +0.25 pp   | LR trop bas → modèle quasi inchangé      |
| ETHICS v2        | ETHICS-train      |  1 000 |   125 | 2e-5   | r=8 q,k,v,o| 0.1  | 0.600     | +6.0 pp    | Premier signal positif (bon LR)          |
| ETHICS v3        | ETHICS-train      |  2 000 |   ?   | 2e-5   | r=8 q,k,v,o| 0.1  | —         | —          | Tué (RTX VRAM dégradait à 425 s/it)      |

(Pour mémoire la run finale, à la racine : ETHICS max — 4 000 paires, 500 steps, r=16+MLP, β = 0.05, LR = 2e-5, macro = **0.710** → **+17 pp**.)

## Mapping fichiers

```
versions_precedentes/
├── README.md                      ← vous êtes ici
├── Compte_rendu_ADL_original.docx version du compte-rendu groupe AVANT injection DPO
├── configs/
│   ├── cpu.yaml                   smoke test CPU (Qwen 0.5B sur 20 ex × 2 cats)
│   ├── dpo.yaml                   PKU-SafeRLHF config originale (T4 fp16)
│   ├── dpo_t4.yaml                variante T4 explicite, fp16 (workaround bug GradScaler bf16)
│   ├── dpo_rtx_lean.yaml          PKU-SafeRLHF en mode lean (RTX, partial step-200)
│   ├── dpo_rtx_ultra.yaml         tentative ultra-lean RTX, jamais menée à terme
│   ├── dpo_ethics.yaml            ETHICS v2 (best config v2)
│   ├── dpo_ethics_v3.yaml         ETHICS v3 (avortée)
│   └── qualitative_bf16.yaml      petit helper pour le script qualitative sur bf16
├── results/
│   ├── baseline_cpu_smoke.json    smoke test CPU (Qwen 0.5B, 2 cats × 20 ex)
│   ├── dpo_eval.json              éval ETHICS de PKU full (T4)
│   ├── dpo_eval_rtx_step200.json  éval ETHICS de PKU step-200
│   ├── dpo_eval_ethics.json       éval ETHICS de v1
│   ├── dpo_eval_ethics_v2.json    éval ETHICS de v2
│   ├── qualitative.json           générations baseline (smoke, baseline-only)
│   └── qualitative_v2.json        générations baseline vs v2
├── logs/                          logs bruts de chaque run (entraînement + éval)
├── notebooks/archive/
│   └── results_analysis_v1.ipynb  version précédente du notebook d'analyse (3 runs seulement)
└── checkpoints/                   adaptateurs LoRA des runs intermédiaires (non versionnés, gitignore)
    ├── dpo_default/checkpoint-200 PKU step-200 (le seul checkpoint partiel encore exploitable)
    ├── dpo_rtx/                   PKU lean RTX (incomplet, abandonné)
    ├── dpo_ethics/checkpoint-125, final  ETHICS v2
    └── dpo_ethics_v3/             ETHICS v3 (abandonné)
```

## Reproduire l'une des runs

Tous les chemins sont relatifs à la **racine du dépôt** (parent de `versions_precedentes/`). Les scripts et le code `src/` n'ont pas changé entre versions.

### Exemple — refaire la run PKU full (T4 Colab, ~4 h)

```bash
python -m scripts.train_dpo \
    --config versions_precedentes/configs/dpo_t4.yaml

python -m scripts.eval_dpo \
    --config versions_precedentes/configs/dpo_t4.yaml \
    --adapter checkpoints/dpo_t4/final \
    --out versions_precedentes/results/dpo_eval_pku_new.json
```

### Exemple — refaire ETHICS v2 (RTX 4070, ~30 min)

```powershell
python -m scripts.train_dpo --config versions_precedentes/configs/dpo_ethics.yaml
python -m scripts.eval_dpo `
    --config versions_precedentes/configs/dpo_ethics.yaml `
    --adapter checkpoints/dpo_ethics/final `
    --out versions_precedentes/results/dpo_eval_ethics_v2_new.json
```

Le notebook `notebooks/results_analysis.ipynb` (à la racine) charge automatiquement les 5 JSON archivés ici en plus du JSON de la run max — il n'y a rien à modifier pour voir le panorama complet des 6 expériences.

## Petite chronologie

1. **PKU full** (Colab T4, ~4 h) — première vraie run. Loss descend sur PKU mais ETHICS dégrade.
2. **PKU step-200** (rejouée localement sur l'adapter sauvé à 200 steps) — early-stopping accidentel.
3. **ETHICS v1** (RTX, LR=5e-6) — premier essai sur dataset in-distribution, n'a pas appris.
4. **ETHICS v2** (RTX, LR=2e-5) — fix LR, premier vrai gain ETHICS (+6 pp).
5. **ETHICS v3** (RTX, double données) — VRAM s'est dégradée, tué avant la fin.
6. **ETHICS max** (RTX, +capacité +β plus permissif) — résultat final +17 pp. **À la racine.**
