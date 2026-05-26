# Projet ADL — Ethical Alignment of Small Language Models

## Contexte général

Projet de M1 Advanced Deep Learning. Objectif : étudier et implémenter des méthodes d'alignement éthique sur de petits modèles de langage (1–3B paramètres), sous contraintes computationnelles réalistes.

Trois approches sont comparées :
- **DPO** — Direct Preference Optimization
- **RLHF** — Reinforcement Learning from Human Feedback
- **RAG-based alignment** — Retrieval-Augmented Generation

Le but n'est pas d'obtenir un modèle parfaitement aligné, mais d'analyser les forces et limites de chaque technique. La métrique commune est la **précision sur le benchmark ETHICS** (Hendrycks et al., ICLR 2021), utilisé uniquement en test (~100 exemples par catégorie : commonsense, justice, etc.).

## Mon périmètre

Je suis responsable de la **partie DPO**.

### Ce que DPO implique
- Fine-tuning du modèle sur des paires de préférences (réponse préférée / réponse rejetée)
- Objectif : augmenter la probabilité des réponses préférées, diminuer celle des rejetées
- Datasets envisageables : Anthropic HH-RLHF, UltraFeedback, OpenAssistant, données synthétiques

## Contraintes techniques

- Modèle de base : open-source, 1–3B paramètres
- Entraînement sur GPU grand public ou serveurs universitaires
- Fine-tuning PEFT fortement recommandé : **LoRA / QLoRA**
- Dataset d'entraînement au choix (pas ETHICS), mais doit être documenté :
  - description du preprocessing
  - mention de données synthétiques éventuelles
  - mention d'LLMs externes utilisés

## Livrables

1. **Rapport** : format papier ACL, 8 pages + références illimitées
   - Introduction & motivation
   - Description des méthodes et datasets
   - Détails d'implémentation & setup expérimental
   - Résultats qualitatifs et quantitatifs
   - Limites & discussion éthique
   - Conclusion
2. **Code source complet**

## Références clés

- Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023
- Ouyang et al., *Training LMs to Follow Instructions with Human Feedback*, NeurIPS 2022
- Hendrycks et al., *Aligning AI With Shared Human Values*, ICLR 2021
- Bai et al., *Constitutional AI: Harmlessness from AI Feedback*, arXiv 2022
