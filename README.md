# DPO — Ethical Alignment de Petits LM (Projet ADL)

Partie **DPO** du projet M1 ADL *Ethical Alignment of Small Language Models*.
Cible : aligner un LLM 1–3B avec **Direct Preference Optimization** (Rafailov et al., NeurIPS 2023),
puis évaluer sur **ETHICS** (Hendrycks et al., ICLR 2021).

---

## 1. Choix de design

| Élément                  | Choix                                 | Justification                                                                                              |
|--------------------------|---------------------------------------|------------------------------------------------------------------------------------------------------------|
| Modèle de base           | `Qwen/Qwen2.5-1.5B-Instruct`          | Apache 2.0, chat template ChatML stable, base instruct correcte → marge DPO observable.                    |
| Méthode d'entraînement   | DPO via TRL                           | Pas de reward model séparé, plus stable à petite échelle que PPO.                                          |
| PEFT                     | QLoRA 4-bit (NF4 + double quant)      | Tient sur GPU grand public (≥ 8 Go VRAM).                                                                  |
| Dataset DPO              | `PKU-Alignment/PKU-SafeRLHF`          | Pairs labellisées explicitement sur l'axe *safety* → signal éthique propre, plus aligné sur ETHICS que HH. |
| Filtrage des paires      | Désaccord sur `is_response_X_safe`    | Élimine les paires où la préférence safety est ambiguë.                                                    |
| Évaluation               | Log-likelihood `P("Yes") vs P("No")`  | Robuste pour 1.5B (génération libre trop bruitée à cette taille).                                          |
| Catégories ETHICS        | commonsense, deontology, justice, virtue | Les 4 tâches binaires ; *utilitarianism* (ranking pairwise) gardée en option future.                    |
| n par catégorie          | 100, équilibré 50/50                  | Conforme à la consigne (~100 ex/catégorie), réduit la variance d'évaluation.                               |

**Données synthétiques :** aucune dans la version par défaut.
**LLMs externes utilisés :** aucun pour la génération de données (DPO sur PKU brut). Claude (cet assistant) est utilisé uniquement pour la scaffolding du code et la rédaction du rapport — pas pour générer des préférences.

---

## 2. Structure du projet

```
prjgourru/
├── configs/dpo.yaml          # tous les hyperparamètres
├── src/
│   ├── data.py               # loaders PKU-SafeRLHF + ETHICS
│   ├── model.py              # base + QLoRA + chargement d'adaptateur
│   ├── train.py              # boucle DPO (TRL DPOTrainer)
│   ├── eval.py               # log-likelihood scoring sur ETHICS
│   └── utils.py              # seed, IO, logging
├── scripts/
│   ├── eval_baseline.py      # éval du modèle non-aligné
│   ├── train_dpo.py          # entraînement DPO unique
│   ├── eval_dpo.py           # éval d'un adaptateur DPO
│   ├── run_ablations.py      # grille (β, lr, taille data)
│   └── qualitative.py        # générations comparées baseline vs DPO
├── results/                  # JSONs d'évaluation (gitignored)
├── checkpoints/              # adaptateurs LoRA (gitignored)
└── requirements.txt
```

---

## 3. Installation

```powershell
# Python 3.10–3.11 recommandé
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows + bitsandbytes :** depuis `0.43.0`, `bitsandbytes` fournit des wheels Windows officiels. Si l'import échoue (CUDA non détecté), exécute :
```powershell
pip install --force-reinstall bitsandbytes
```
En dernier recours, désactive QLoRA en passant `model.load_in_4bit: false` dans `configs/dpo.yaml` (il faudra alors plus de VRAM ou un modèle plus petit).

**Auth Hugging Face :**
```powershell
huggingface-cli login
```
Qwen2.5 et PKU-SafeRLHF sont publics ; ETHICS aussi.

---

## 4. Pipeline complet

```powershell
# 1. Baseline — pas d'entraînement, juste l'éval initiale
python -m scripts.eval_baseline --config configs/dpo.yaml --out results/baseline_eval.json

# 2. Entraînement DPO principal
python -m scripts.train_dpo --config configs/dpo.yaml

# 3. Évaluation du modèle DPO
python -m scripts.eval_dpo `
    --config configs/dpo.yaml `
    --adapter checkpoints/dpo_default/final `
    --out results/dpo_eval.json

# 4. Analyse qualitative (générations cote à cote)
python -m scripts.qualitative `
    --config configs/dpo.yaml `
    --adapter checkpoints/dpo_default/final `
    --out results/qualitative.json

# 5. Ablations (β, LR, taille data) — long
python -m scripts.run_ablations --config configs/dpo.yaml --out results/ablations.json
```

---

## 5. Méthodologie détaillée

### 5.1 Construction des paires DPO

PKU-SafeRLHF fournit pour chaque prompt deux réponses `response_0`, `response_1` et plusieurs labels. On utilise :

- `safer_response_id` ∈ {0, 1} pour orienter la paire sur l'axe **safety**, *pas* `better_response_id` (qui mélange helpfulness et safety) ;
- on **filtre** sur `is_response_0_safe ≠ is_response_1_safe` pour ne garder que les paires avec un contraste de sûreté clair.

Le prompt est ensuite formaté avec le chat template Qwen (ChatML) avant d'être passé à `DPOTrainer`.

### 5.2 Objectif DPO

Notation : `π_θ` = policy en cours d'entraînement, `π_ref` = policy de référence (le modèle de base figé). DPO optimise :

```
L_DPO(θ) = -E[ log σ ( β · ( log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]
```

où `y_w` = chosen (réponse safe), `y_l` = rejected. Avec PEFT, la policy de référence est obtenue gratuitement en désactivant les adaptateurs LoRA — pas besoin de charger un deuxième modèle en mémoire.

### 5.3 Évaluation ETHICS

Pour chaque exemple binaire, on construit un prompt yes/no et on calcule, sous le chat template :

```
logP(answer = "Yes" | prompt)   et   logP(answer = "No" | prompt)
```

en sommant les log-probabilités au niveau token. Prédiction = `argmax`. Acc = #correct / N.

Mapping des labels :

| Catégorie    | label 1 ↔ réponse "Yes" attendue                            |
|--------------|--------------------------------------------------------------|
| commonsense  | l'action est moralement *wrong*                              |
| deontology   | l'excuse est *reasonable*                                    |
| justice      | la revendication est *reasonable*                            |
| virtue       | le trait *correspond* à la situation                         |

### 5.4 Ablations prévues

Trois axes, à exécuter selon le budget GPU :

- **β ∈ {0.05, 0.1, 0.3}** — contrôle la régularisation KL vs `π_ref`. β petit = plus de drift.
- **learning rate ∈ {1e-6, 5e-6, 2e-5}** — sensibilité usuelle DPO.
- **taille du training set ∈ {5k, 20k}** — diminishing returns ?

Pour rester dans un budget raisonnable (~3 GPU-h sur un seul GPU 12 Go) on fixe deux axes et on varie un seul.

---

## 6. Métriques rapportées dans le rapport

Pour chaque modèle (baseline + chaque run DPO) :

- accuracy par catégorie ETHICS,
- macro-moyenne sur les 4 catégories,
- delta vs baseline,
- 10 générations qualitatives sur les exemples où le baseline échoue,
- 5 générations sur les *safety probes* de `scripts/qualitative.py`.

Tableaux et figures (matplotlib) à générer à partir des JSON dans `results/`.

---

## 7. Limites connues à discuter dans la section "Limites"

1. **Évaluation par log-likelihood ≠ déploiement réel** : un modèle peut bien classifier "Yes/No" sans pour autant générer des réponses sûres. La section qualitative compense partiellement.
2. **Train/test domain gap** : PKU-SafeRLHF = dialogue, ETHICS = scénarios courts. Le transfer est partiel.
3. **Petite taille du modèle (1.5B)** : performances absolues modestes ; les conclusions portent sur le *delta* baseline → DPO, pas sur l'alignement absolu.
4. **n = 100 / catégorie** : intervalle de confiance ~±10 points à 80% acc. À rapporter dans le tableau.
5. **PKU-SafeRLHF est lui-même biaisé** (annotateurs, distribution de prompts) — héritage de biais à mentionner.

---

## 8. Pour le rapport ACL 8 pages

Plan suggéré :

1. **Intro & motivation** (¾ page) — pourquoi aligner les petits LM, focus DPO.
2. **Background** (1 page) — DPO formel, ETHICS.
3. **Méthode** (1.5 pages) — choix de PKU + filtrage, QLoRA, prompt format ETHICS.
4. **Setup expérimental** (1 page) — modèle, hyperparams, hardware.
5. **Résultats quantitatifs** (1.5 pages) — tableau baseline vs DPO, courbe d'ablation β.
6. **Résultats qualitatifs** (1 page) — 3–4 exemples commentés.
7. **Limites & éthique** (½ page) — cf. section 7.
8. **Conclusion** (¼ page).

---

## 9. Référencement

- Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023.
- Hendrycks et al., *Aligning AI With Shared Human Values*, ICLR 2021.
- Ji et al., *PKU-SafeRLHF*, NeurIPS 2024.
- Hu et al., *LoRA*, ICLR 2022.
- Dettmers et al., *QLoRA*, NeurIPS 2023.
