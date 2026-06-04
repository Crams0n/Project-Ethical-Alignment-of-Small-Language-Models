"""Inject the DPO section into the team's Compte_rendu_ADL.docx.

Strategy:
- Detect the [À COMPLÉTER] paragraphs (red text) by exact substring match.
- Replace their content with the DPO writeup (3-4 pages total).
- Fill the empty DPO row in Table 0.
- Insert 3 figures (fig_dpo_macro / fig_dpo_categories / fig_dpo_delta) at coherent positions.
- Save to Compte_rendu_ADL.docx (the original is backed up before any modification).
"""
from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, RGBColor, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = ROOT / "Compte_rendu_ADL.docx"
DOCX_BACKUP = ROOT / "Compte_rendu_ADL_original.docx"
RESULTS = ROOT / "results"
FIG_MACRO = RESULTS / "fig_dpo_macro.png"
FIG_CATEGORIES = RESULTS / "fig_dpo_categories.png"
FIG_DELTA = RESULTS / "fig_dpo_delta.png"


# ----- DPO writeup content (matches the existing voice/structure) -----------

DPO_22 = (
    "Notre implémentation s'appuie sur la bibliothèque TRL (DPOTrainer) avec une politique "
    "de référence π_ref obtenue gratuitement, en désactivant les adaptateurs LoRA sur le même "
    "modèle de base (un seul jeu de poids résident en VRAM). Nous explorons deux familles "
    "de préférences. (a) PKU-SafeRLHF (Ji et al., 2024) : préférences générées par annotation "
    "humaine sur des dialogues de safety ; nous filtrons les paires où is_response_0_safe ≠ "
    "is_response_1_safe afin d'isoler un signal sécurité non ambigu (10 796 paires conservées "
    "sur ~330 k). (b) Préférences synthétiques dérivées d'ETHICS train : pour chaque exemple, "
    "on reformate le scénario sous le même template Yes/No que l'évaluation, chosen = réponse "
    "gold et rejected = son opposée. Configuration commune : QLoRA 4-bit NF4 + double "
    "quantification, LoRA r ∈ {8, 16} selon la run, modules cibles {q,k,v,o}_proj et — pour "
    "la configuration optimale — {gate,up,down}_proj, optimiseur paged_adamw_8bit, gradient "
    "checkpointing, bf16 sur GPU Ada (RTX 4070 Laptop) ou fp16 sur Turing (T4 Colab)."
)

DPO_23 = (
    "Données synthétiques et LLM externe. Le jeu de préférences ETHICS-train est généré par "
    "une transformation purement déterministe (étiquette gold → chosen, étiquette opposée → "
    "rejected) appliquée aux exemples train d'ETHICS — jamais sur le test. Aucun LLM externe "
    "n'est utilisé pour produire ni filtrer les préférences. Cette transparence garantit "
    "l'absence de contamination test à test."
)

DPO_40 = (
    "Volet DPO. Deux sources de préférences sont comparées. (i) PKU-Alignment/PKU-SafeRLHF "
    "(Ji et al., 2024), filtré sur le désaccord d'annotation de safety (10 796 paires sur "
    "~330 k). (ii) Préférences synthétiques issues d'ETHICS train (4 catégories × jusqu'à "
    "1 000 exemples = 4 000 paires) construites par transformation déterministe (chosen = "
    "label gold, rejected = label opposé). Aucune donnée test d'ETHICS n'est utilisée à "
    "l'entraînement. Aucun LLM externe n'intervient dans la construction des préférences."
)

DPO_48 = (
    "Setup DPO (configuration optimale, dite « max »). QLoRA 4-bit NF4 + double quantification, "
    "LoRA r = 16 / α = 32 / dropout = 0.05 sur {q,k,v,o,gate,up,down}_proj (≈ 12 M paramètres "
    "trainables, soit 0,8 % des 1,5 G). DPO β = 0.05, learning rate = 2e-5 (cosine, warmup "
    "10 %), batch effectif 8 (per_device = 1 + grad_accum = 8), 500 pas d'optimisation sur "
    "4 000 paires (1 époque), seed = 42, optimiseur paged_adamw_8bit. Matériel : NVIDIA RTX "
    "4070 Laptop (8 Go VRAM), bf16 natif (Ada). Temps mesurés : ~3 h training + ~3 min "
    "évaluation ETHICS (4 cats × 100 ex × 2 forward Yes/No). Trois autres configurations ont "
    "été entraînées comme ablation (voir §4.1.b)."
)

DPO_76 = (
    "Volet DPO. Inspection libre des générations produites par le modèle aligné (configuration "
    "max) sur deux familles de prompts. (a) Scénarios commonsense ETHICS où le baseline se "
    "trompait : sur 7 cas inspectés, la politique alignée corrige 5 décisions en génération "
    "libre — notamment « I killed the kitten in my bed » (No → Yes), « I planted one on him "
    "to see if sparks would fly » (No → Yes) et « AITA defending my best man to my fiancée » "
    "(implicitement → Yes). Les 2 décisions qui restent inchangées sont les phrases brèves et "
    "lapidaires (« I smiled when the girl was crying », « I teased Kelly's transition »), où "
    "la distribution résiduelle du sampling tire encore vers le baseline. (b) Cinq safety "
    "probes hand-written. Sur les deux prompts d'intention nuisible explicite (impersonation "
    "bancaire, demande de méthode discrète pour blesser), la politique alignée maintient le "
    "refus du baseline et ajoute une redirection vers une ressource d'aide (ami / famille / "
    "professionnel de santé mentale). Sur les prompts à valeur éthique ambivalente "
    "(« est-il acceptable de mentir pour épargner autrui », « faut-il signaler un collègue qui "
    "vole de petites fournitures »), elle adopte un ton plus engagé : moins d'hyperboles "
    "factuelles (le baseline affirmait que mentir est « illégal dans beaucoup de pays »), plus "
    "de structuration en points. Conclusion qualitative : le gain d'accuracy en yes/no scoring "
    "ne se paie pas par une régression du comportement free-form."
)

DPO_88 = (
    "Côté DPO, le résultat principal est que l'algorithme fonctionne comme prévu sur son "
    "propre objectif, mais que la source des préférences est de loin le facteur dominant. "
    "Entraîné sur PKU-SafeRLHF (préférences de safety en dialogue), DPO converge nettement "
    "(rewards/accuracies = 0,876 sur le split eval interne) mais dégrade ETHICS de 1,5 pp "
    "macro, avec un effondrement du rappel sur la classe « yes » sur virtue (−10 pp). "
    "Substituer les préférences par des paires synthétiques issues d'ETHICS train (même "
    "algorithme, mêmes hyper-paramètres principaux, simple changement de dataset) inverse "
    "complètement le signe : +6,0 pp macro avec une configuration minimale, puis +17,0 pp "
    "macro (54,0 % → 71,0 %) en ajoutant capacité (LoRA r 8 → 16 + modules MLP), volume "
    "(1 000 → 4 000 paires) et drift autorisé (β 0,1 → 0,05). Les gains sont cohérents sur "
    "les quatre sous-ensembles : +13 pp commonsense, +24 pp deontology, +20 pp justice, "
    "+11 pp virtue. La leçon pratique côté DPO : tenter de récupérer d'un dataset de "
    "préférences mal aligné par un balayage d'hyper-paramètres est un cul-de-sac ; le "
    "levier qui a déterminé notre score final est la substitution du dataset, réalisée une "
    "fois, et tout le reste (rang LoRA, volume, β, learning rate) en a été un multiplicateur. "
    "[À COMPLÉTER PAR LE GROUPE — comparaison des trois approches (performance, coût, "
    "auditabilité) et choix de la meilleure stratégie selon les contraintes.]"
)

# Body paragraphs to insert in §4.1, after the existing RAG/RLHF discussion ----

DPO_RESULTS_INTRO = (
    "Volet DPO. Le tableau ci-dessus inclut la meilleure configuration DPO obtenue (run "
    "« max », ligne dédiée), évaluée par son propre harnais : RTX 4070 Laptop en bf16, "
    "scoring log-probabilité Yes/No, 100 exemples par sous-ensemble, 4 sous-ensembles "
    "(commonsense, deontology, justice, virtue ; utilitarianism n'est pas binaire et "
    "demanderait un autre format de scoring que nous n'avons pas développé). Son baseline "
    "interne (Qwen2.5-1.5B sans alignement, même harnais) vaut 54,0 % — différent du baseline "
    "joint à 59,0 % à cause d'un template de prompt légèrement différent. Comme pour la ligne "
    "RLHF, seul l'écart DPO − baseline interne est interprétable directement : +17,0 pp macro."
)

DPO_FIG1_CAPTION = (
    "Figure 4 — Six configurations DPO comparées sur la macro-accuracy ETHICS. La barre grise "
    "(baseline 0,540) sert de référence. Les barres rouges sont entraînées sur PKU-SafeRLHF "
    "(safety générique) ; les barres vertes sur des préférences synthétiques d'ETHICS train. "
    "Le passage du rouge au vert marque le changement de dataset ; la profondeur du vert "
    "reflète l'augmentation de capacité (rang LoRA, volume de données, β plus permissif)."
)

DPO_ABLATION_TITLE = "4.1.b. Ablation DPO : six configurations"

DPO_ABLATION_BODY = (
    "Pour isoler ce qui fait la différence côté DPO, nous avons entraîné six configurations "
    "(figure 4 ci-dessus). Trois observations s'en dégagent. Premièrement, le dataset domine "
    "la régularisation : à β = 0,1 et avec la même politique de référence, PKU-SafeRLHF en "
    "régime complet (1 350 pas, 10,8 k paires) baisse la macro de 1,5 pp, tandis que son "
    "early-stop à 200 pas la relève de 1,0 pp — autrement dit, plus on entraîne sur PKU, "
    "moins ETHICS progresse. Deuxièmement, le learning rate adapté à un signal mono-token "
    "compte : sur ETHICS-train, passer de 5e-6 à 2e-5 fait basculer le résultat de +0,25 pp "
    "(quasi pas appris) à +6,0 pp (run « v2 »). Troisièmement, une fois le bon dataset et le "
    "bon LR fixés, les leviers classiques de scaling (rang LoRA × 2, modules MLP ajoutés, "
    "données × 4, β / 2) se composent multiplicativement pour atteindre +17,0 pp macro — sans "
    "qu'aucune ablation séparée n'aurait pu produire ce gain seule."
)

DPO_FIG3_CAPTION = (
    "Figure 5 — Δ accuracy par sous-ensemble vs baseline, pour deux configurations DPO qui "
    "ne diffèrent que par le dataset de préférences. À gauche (rouge) : PKU-SafeRLHF (safety "
    "dialoguale, hors distribution). À droite (vert) : préférences synthétiques d'ETHICS train "
    "(en distribution). Les deux runs partagent le même algorithme et tournent toutes deux sur "
    "Qwen2.5-1.5B + QLoRA, mais leur signe est opposé sur les quatre sous-ensembles."
)

DPO_FIG2_CAPTION = (
    "Figure 6 — Accuracy par sous-ensemble : baseline Qwen2.5-1.5B vs notre meilleure "
    "configuration DPO (ETHICS-train, r=16+MLP, β=0,05, 4 000 paires, 500 pas). Les gains "
    "vont de +11 pp (virtue) à +24 pp (deontology), tous positifs et tous au-dessus du seuil "
    "du hasard."
)


# ----- DPO row in the table (R3) -------------------------------------------

DPO_ROW_VALUES = ["DPO (ETHICS-train, max)", "63.0", "76.0", "71.0", "74.0", "—", "71.0"]
DPO_ROW_FOOTNOTE = (
    "* Volet DPO évalué par son propre harnais (4 sous-ensembles binaires × 100 ex, scoring "
    "log-probabilité, RTX 4070 bf16). Son baseline interne vaut 54.0 et non 59.0. Comme pour "
    "RLHF, seul l'écart DPO − baseline interne (+17 pp macro) est interprétable directement."
)


# ----- Helpers --------------------------------------------------------------

def find_paragraph_by_substring(doc, substring: str) -> Paragraph | None:
    for p in doc.paragraphs:
        if substring in p.text:
            return p
    return None


def replace_paragraph_text(p: Paragraph, new_text: str) -> None:
    """Drop all runs, then write a single fresh run with default formatting."""
    for run in list(p.runs):
        run._element.getparent().remove(run._element)
    new_run = p.add_run(new_text)
    new_run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    new_run.font.bold = False
    new_run.font.italic = False


def insert_paragraph_after(paragraph: Paragraph, text: str = "", style_name: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._element.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style_name:
        new_para.style = paragraph.part.document.styles[style_name]
    if text:
        new_para.add_run(text)
    return new_para


def insert_image_after(paragraph: Paragraph, image_path: Path, width_inches: float = 6.0) -> Paragraph:
    new_para = insert_paragraph_after(paragraph)
    run = new_para.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    new_para.alignment = 1  # center
    return new_para


# ----- Main update ----------------------------------------------------------

def main():
    if not DOCX_BACKUP.exists():
        shutil.copy2(DOCX_PATH, DOCX_BACKUP)
        print(f"Backed up original to {DOCX_BACKUP.name}")

    doc = Document(str(DOCX_PATH))

    # 1) Replace red-text placeholders with DPO content
    replacements = [
        ("[À COMPLÉTER — décrire le jeu de préférences", DPO_22),
        ("[À COMPLÉTER — préciser si des données synthétiques", DPO_23),
        ("[À COMPLÉTER — DPO — documenter ici le jeu d'entraînement", DPO_40),
        ("[À COMPLÉTER — DPO — préciser le setup", DPO_48),
        ("[À COMPLÉTER — DPO — ajouter une analyse qualitative", DPO_76),
        ("[À COMPLÉTER — synthétiser les résultats DPO", DPO_88),
    ]
    for needle, new_text in replacements:
        p = find_paragraph_by_substring(doc, needle)
        if p is None:
            print(f"WARN: placeholder not found: {needle[:50]}")
            continue
        replace_paragraph_text(p, new_text)
        print(f"Replaced: {needle[:50]}...")

    # 2) Fill the DPO row in Table 0
    table = doc.tables[0]
    dpo_row = table.rows[3]
    for cell, value in zip(dpo_row.cells, DPO_ROW_VALUES):
        cell.text = value  # this clears and resets
    # update footnote paragraph immediately after the table
    foot = find_paragraph_by_substring(doc, "* Volet RLHF évalué par son propre pipeline")
    if foot is not None:
        new_foot = foot.text + "\n" + DPO_ROW_FOOTNOTE
        # Append the DPO footnote as a new paragraph just after the RLHF one
        insert_paragraph_after(foot, DPO_ROW_FOOTNOTE, style_name=foot.style.name)
        print("Added DPO footnote after RLHF footnote.")

    # 3) Insert DPO intro paragraph + Figure 4 (macro) + ablation subsection
    # We anchor after the RLHF lecture/interpretation paragraph (just before §4.2 heading).
    anchor_4_2 = find_paragraph_by_substring(doc, "4.2. Résultats CoT et ablation")
    if anchor_4_2 is None:
        # fallback: insert before §4.3 (qualitative)
        anchor_4_2 = find_paragraph_by_substring(doc, "4.3. Analyse qualitative")
    if anchor_4_2 is not None:
        # We insert BEFORE the section heading. python-docx has no insert_before,
        # so we insert AFTER the previous sibling. Easiest: insert after the
        # RLHF interpretation paragraph that ends with "Pistes futures" or
        # "Lecture. Ce résultat..."
        rlhf_interp = find_paragraph_by_substring(doc, "Lecture. Ce résultat, décevant en apparence")
        if rlhf_interp is None:
            rlhf_interp = anchor_4_2
            # In that case we insert AFTER the heading itself, which still places
            # the DPO block before §4.2 body
        cursor = rlhf_interp
        cursor = insert_paragraph_after(cursor, DPO_RESULTS_INTRO)
        cursor = insert_image_after(cursor, FIG_MACRO, width_inches=6.2)
        cursor = insert_paragraph_after(cursor, DPO_FIG1_CAPTION)
        # Subsection 4.1.b heading
        cursor = insert_paragraph_after(cursor, DPO_ABLATION_TITLE, style_name="Heading 2")
        cursor = insert_paragraph_after(cursor, DPO_ABLATION_BODY)
        cursor = insert_image_after(cursor, FIG_DELTA, width_inches=6.2)
        cursor = insert_paragraph_after(cursor, DPO_FIG3_CAPTION)
        cursor = insert_image_after(cursor, FIG_CATEGORIES, width_inches=6.2)
        cursor = insert_paragraph_after(cursor, DPO_FIG2_CAPTION)
        print("Inserted DPO §4.1.b block with 3 figures.")
    else:
        print("WARN: could not anchor §4.1.b before §4.2.")

    doc.save(str(DOCX_PATH))
    print(f"\nSaved {DOCX_PATH.name}")


if __name__ == "__main__":
    main()
