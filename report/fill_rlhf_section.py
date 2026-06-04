# -*- coding: utf-8 -*-
import copy
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC='/home/ubuntu/workspace/gdrive/ethical-alignment-rl/Compte_rendu_ADL.docx'
OUT='/tmp/Compte_rendu_ADL.docx'
d=docx.Document(SRC)
paras=d.paragraphs

def find(prefix, contains=None):
    for p in d.paragraphs:
        t=p.text.strip()
        if t.startswith(prefix) and (contains is None or contains in t):
            return p
    raise SystemExit(f"not found: {prefix!r}")

def style_runs(p, segments):
    # segments: list of (text, bold)
    for txt,bold in segments:
        r=p.add_run(txt)
        r.bold=bold
        r.font.size=Pt(11)

def set_para(p, segments):
    # clear existing runs
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    style_runs(p, segments)

def insert_before(anchor, segments, style='Normal', italic=False, align=None):
    p=anchor.insert_paragraph_before(style=style)
    if align is not None: p.alignment=align
    for txt,bold in segments:
        r=p.add_run(txt); r.bold=bold; r.italic=italic; r.font.size=Pt(11 if not italic else 9)
    return p

def insert_fig_before(anchor, path, width):
    p=anchor.insert_paragraph_before()
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    return p

def caption_before(anchor, text):
    p=anchor.insert_paragraph_before()
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run(text); r.italic=True; r.font.size=Pt(9)
    r.font.color.rgb=RGBColor(0x44,0x44,0x44)
    return p

# ============ 2.3 RLHF — méthode ============
p26=find("[À COMPLÉTER — décrire le jeu de données de récompense")
set_para(p26,[
 ("Notre implémentation suit le pipeline en trois temps de Ouyang et al. (2022), "
  "dont nous omettons volontairement la première étape (fine-tuning supervisé) : le "
  "modèle de base Qwen2.5-1.5B-Instruct étant déjà instruction-tuné, il en tient lieu. "
  "Nous nous concentrons donc sur (2) l'apprentissage d'un modèle de récompense et "
  "(3) l'optimisation de la politique, tous deux construits sur le même "
  "Qwen2.5-1.5B-Instruct que les autres approches, par souci de cohérence de la comparaison.",False),
])
# new paragraph (reward model architecture) inserted right after p26, before p27
p27=find("[À COMPLÉTER — indiquer le prétraitement, l'usage éventuel de données synthét")
insert_before(p27,[
 ("Modèle de récompense. ",True),
 ("Nous ajoutons au modèle de base une tête de classification scalaire (une seule sortie, "
  "AutoModelForSequenceClassification, num_labels=1) qui prédit un score de qualité. Sous "
  "contrainte d'un unique GPU T4 (16 Go), le modèle est chargé en 4 bits (QLoRA : "
  "quantification NF4, calcul en bfloat16, double quantification) et seuls des adaptateurs "
  "LoRA (rang r=16, α=32, dropout=0.05) insérés sur les projections d'attention q, k, v et o "
  "sont entraînés — soit environ 9 M de paramètres entraînables, ~0,6 % du modèle. "
  "L'entraînement utilise le RewardTrainer de TRL, qui minimise la perte de Bradley-Terry "
  "−log σ(r_choisi − r_rejeté) : pour chaque paire (réponse préférée, réponse rejetée), le "
  "modèle apprend à attribuer un score supérieur à la réponse préférée par l'humain. "
  "Configuration : 1 époque, batch effectif 16 (4 × accumulation 4), learning rate 5·10⁻⁵, "
  "longueur maximale 512 tokens, graine 42.",False),
])
# replace p27 with RLOO description
set_para(p27,[
 ("Optimisation de la politique : RLOO plutôt que PPO. ",True),
 ("La consigne autorise explicitement une « alternative plus légère à PPO ». Nous retenons "
  "RLOO (REINFORCE Leave-One-Out, Ahmadian et al., 2024) pour deux raisons concrètes. "
  "D'abord, la version de TRL utilisée (1.5) a retiré le PPOTrainer historique. Ensuite, RLOO "
  "supprime le réseau de valeur (value head) et le GAE de PPO : pour chaque prompt il "
  "échantillonne K complétions et utilise la récompense moyenne des K−1 autres comme ligne "
  "de base, ce qui réduit fortement l'empreinte mémoire — décisif sur un T4. Une copie gelée "
  "de la politique sert de référence pour une pénalité KL (β=0.04) qui empêche le modèle de "
  "trop s'éloigner de son point de départ (garde-fou contre le reward hacking). "
  "Configuration RLOO : K=4 générations par prompt, 500 prompts, 1 époque, learning rate "
  "1·10⁻⁶ (volontairement faible pour la stabilité), batch 4 × accumulation 2, β_KL=0.04, "
  "température 1.0, complétions limitées à 64 tokens (prompts à 256).",False),
])

# ============ 2.5 — données RLHF ============
p38=find("[À COMPLÉTER — DPO/RLHF — documenter ici les jeux d'entraînement")
insert_before(p38,[
 ("Volet RLHF : ",True),
 ("nous utilisons le jeu de préférences Anthropic/HH-RLHF (Bai et al., 2022), constitué de "
  "dialogues où deux réponses d'assistant sont comparées, l'une « choisie » et l'autre "
  "« rejetée » par un annotateur humain selon des critères d'utilité et d'innocuité (helpful & "
  "harmless). Prétraitement : 3 000 paires d'entraînement et 300 d'évaluation, formatées via le "
  "chat template de Qwen et tronquées à 512 tokens ; les mêmes prompts (sans les réponses) "
  "fournissent les 500 invites de l'étape RLOO. Aucune donnée synthétique n'a été générée et "
  "aucun LLM externe n'a été employé : le modèle de récompense, la politique et la référence "
  "proviennent tous du seul Qwen2.5-1.5B-Instruct, et les préférences sont d'origine humaine. "
  "Aucune fuite vers ETHICS n'est possible : HH-RLHF et ETHICS sont indépendants et ETHICS "
  "n'est utilisé qu'en test.",False),
])
set_para(p38,[("[À COMPLÉTER — DPO — documenter ici le jeu d'entraînement : source, étapes de "
 "prétraitement, génération éventuelle de données synthétiques, et usage éventuel de LLM "
 "externes (exigé par la consigne, section 4).]",False)])

# ============ 3.2 — setup RLHF ============
p45=find("[À COMPLÉTER — DPO/RLHF — préciser le setup")
insert_before(p45,[
 ("Setup RLHF. ",True),
 ("Tout l'entraînement tient sur un unique GPU T4 16 Go (Colab/Kaggle gratuit) grâce à QLoRA "
  "4 bits + LoRA (voir 2.3). Optimiseur AdamW, graine 42. Temps de calcul observés : ~1 h pour "
  "le modèle de récompense (3 000 paires, 174 pas), ~2 h pour l'optimisation RLOO (500 prompts "
  "× 4 générations, 249 pas) et ~25 min pour l'évaluation comparative baseline vs RLHF. "
  "L'ensemble est « checkpointé » (reprise après déconnexion) et écrit ses sorties — "
  "adaptateurs, métriques et figures — dans un dossier partagé. Précision importante : pour "
  "tenir dans le budget de temps du T4, l'évaluation ETHICS du volet RLHF a été menée sur "
  "50 exemples par sous-ensemble (au lieu de 100), avec son propre baseline mesuré dans "
  "exactement les mêmes conditions.",False),
])
set_para(p45,[("[À COMPLÉTER — DPO — préciser le setup (LoRA/QLoRA, taille de batch, optimiseur, "
 "matériel) et le temps d'entraînement.]",False)])

# ============ Tableau 4.1 — ligne RLHF ============
tab=d.tables[0]
row=tab.rows[4].cells
vals=["RLHF*","62.0","50.0","62.0","80.0","58.0","62.4"]
for c,v in zip(row,vals):
    c.text=""
    rr=c.paragraphs[0].add_run(v)
    if v=="RLHF*": rr.bold=False

# note étoile sous le tableau (après la paragraphe "Note : CS = ...")
note_p=find("Note : CS = commonsense")
new=note_p.insert_paragraph_before  # placeholder
# insert AFTER note_p using addnext
np=copy.deepcopy(note_p._p)
# build a fresh paragraph after note_p
from docx.text.paragraph import Paragraph
new_el=note_p._p.makeelement(note_p._p.tag, {})
note_p._p.addnext(new_el)
star=Paragraph(new_el, note_p._parent)
r=star.add_run("* Volet RLHF évalué par son propre pipeline (50 exemples/sous-ensemble, "
 "harnais distinct des lignes Baseline/RAG ci-dessus) : son baseline interne vaut 62.0 et "
 "non 59.0. Seul l'écart RLHF − baseline interne est interprétable : +0,4 pt en moyenne, "
 "porté par le seul utilitarianism (56 → 58).")
r.italic=True; r.font.size=Pt(9); r.font.color.rgb=RGBColor(0x44,0x44,0x44)

# ============ 4.1 — Résultats RLHF + figures (avant le titre 4.2) ============
h42=find("4.2. Résultats CoT et ablation")
insert_before(h42,[("Résultats du volet RLHF.",True)])
insert_before(h42,[
 ("Modèle de récompense. ",True),
 ("La figure ci-dessous montre l'apprentissage du modèle de récompense. La perte de "
  "validation passe sous la barre du hasard (ln 2 ≈ 0,693) et se stabilise autour de 0,688, "
  "tandis que l'exactitude des préférences sur le jeu de validation atteint ~0,557 (jusqu'à "
  "0,567). Le signal appris est donc réel mais faible : entraîné 1 époque en LoRA 4 bits sur "
  "3 000 paires, le modèle ne distingue la bonne réponse que dans ~56 % des cas (marge moyenne "
  "≈ +0,05). C'est un plafond déterminant — la qualité de la récompense borne mécaniquement ce "
  "que l'étape RL pourra exploiter.",False),
])
insert_fig_before(h42,'/tmp/fig_reward.png',6.4)
caption_before(h42,"Figure 1 — Entraînement du modèle de récompense (Qwen2.5-1.5B + LoRA, "
 "HH-RLHF, 1 époque) : perte (gauche) et exactitude des préférences (droite). Les carrés "
 "rouges sont les points de validation.")
insert_before(h42,[
 ("Optimisation RLOO. ",True),
 ("Pendant l'entraînement de la politique, la récompense moyenne par batch reste élevée "
  "(≈ 5,6 en moyenne) mais ne progresse pas nettement ; la divergence KL à la politique de "
  "référence demeure minuscule (de l'ordre de 10⁻³ par token) et l'entropie stable (~1,2). "
  "Autrement dit, la politique bouge très peu : avec un signal de récompense bruité et une "
  "pénalité KL active, RLOO se limite à des ajustements marginaux plutôt qu'à une réécriture "
  "du comportement — prudent et attendu, mais cela borne l'effet mesurable.",False),
])
insert_fig_before(h42,'/tmp/fig_rloo.png',6.4)
caption_before(h42,"Figure 2 — Optimisation de la politique RLOO (K=4, β=0.04, lr=1e-6, "
 "500 prompts HH-RLHF) : récompense des complétions (gauche), divergence KL et entropie "
 "(droite).")
insert_before(h42,[
 ("Évaluation sur ETHICS. ",True),
 ("La figure compare, sous-ensemble par sous-ensemble, le modèle de base et la politique "
  "alignée par RLHF. Les deux séries se superposent presque parfaitement : l'exactitude "
  "moyenne passe de 62,0 % à 62,4 % (+0,4 pt) et le seul sous-ensemble qui progresse est "
  "l'utilitarianism (56 → 58 %, +2 pts) ; tous les autres sont inchangés. Le RLHF tel "
  "qu'implémenté n'améliore donc pas, de façon mesurable, la classification éthique "
  "(voir la ligne RLHF du tableau ci-dessus).",False),
])
insert_fig_before(h42,'/home/ubuntu/workspace/gdrive/ethical-alignment-rl/outputs/eval/comparison.png',6.0)
caption_before(h42,"Figure 3 — Exactitude ETHICS par sous-ensemble : modèle de base vs "
 "politique alignée RLHF (50 exemples/sous-ensemble). La ligne pointillée marque le hasard.")
insert_before(h42,[
 ("Lecture. ",True),
 ("Ce résultat, décevant en apparence, est instructif et cohérent avec la chaîne de causes "
  "ci-dessus : (i) le modèle de récompense n'apprend qu'un signal faible (~56 % d'exactitude "
  "de préférence) ; (ii) la pénalité KL et un learning rate bas maintiennent la politique près "
  "de son point de départ ; (iii) surtout, les préférences HH-RLHF portent sur l'utilité et "
  "l'innocuité conversationnelles, objectif assez éloigné de la classification morale d'ETHICS "
  "(décalage de distribution). Comme pour le RAG, on retrouve le constat central du projet : "
  "sur un petit modèle déjà instruction-tuné, une couche d'alignement supplémentaire n'apporte "
  "qu'un gain marginal, et la valeur du RLHF tient surtout à la maîtrise du pipeline "
  "— récompense apprise puis politique optimisée — qu'il permet de démontrer.",False),
])

# ============ 4.3 — qualitatif RLHF ============
p60=find("[À COMPLÉTER — DPO/RLHF — ajouter une analyse qualitative")
insert_before(p60,[
 ("Volet RLHF. ",True),
 ("Avec le scoring par log-probabilité (section 3.2), les décisions du modèle de base et de "
  "la politique RLHF sont, dans l'immense majorité des cas, identiques : sur les sous-ensembles "
  "où l'exactitude ne bouge pas (commonsense, deontology, justice, virtue), la politique "
  "alignée prend exactement les mêmes décisions que le modèle de départ — confirmation directe "
  "du très faible déplacement mesuré par la KL. Les rares différences se concentrent sur "
  "l'utilitarianism, où la politique tranche correctement quelques scénarios de comparaison de "
  "bien-être que le modèle de base classait à l'envers. Cette quasi-identité illustre "
  "concrètement la prudence de RLOO sous forte régularisation KL : le modèle préserve ses "
  "acquis plutôt que de risquer une dégradation.",False),
])
set_para(p60,[("[À COMPLÉTER — DPO — ajouter une analyse qualitative (exemples de réponses "
 "avant/après alignement, cas de réussite et d'échec).]",False)])

d.save(OUT)
print("saved", OUT)
