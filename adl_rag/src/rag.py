"""
RAG-based alignment
===================

Constitutional-AI-style alignment without any fine-tuning: at inference
time we retrieve the top-K most relevant ethical principles from a
curated corpus and inject them into the system prompt.

The corpus is a hand-written "ethical constitution" of ~50 principles
organised into five families that mirror the ETHICS subsets. Retrieval is
*routed*: each subset queries only the families relevant to its decision
(see config.SUBSET_TO_FAMILIES), which sharply raises the topical relevance
of what gets injected. The families are:

    UNIVERSAL  ── core moral common ground (harm, dignity, honesty)
    JUSTICE    ── what people are owed; fairness, desert
    DEONTOLOGY ── duties, rights, consent, promise-keeping
    VIRTUE     ── character traits and their exemplars
    UTILITY    ── consequence-weighing; wellbeing arithmetic

The stratification is not decorative: it lets us run a family-removal
ablation that proves the retriever is doing the right thing.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

from . import config as C


# ════════════════════════════════════════════════════════════════════════════
#  Hand-curated ethical constitution
# ════════════════════════════════════════════════════════════════════════════

CORPUS: List[dict] = [

    # ─── UNIVERSAL ─────────────────────────────────────────────────────────
    # Core moral common ground. Drawn on by every subset (especially the
    # broad "commonsense morality" one).
    {"family": "universal",
     "text": "Do not cause unnecessary harm to any person, animal, or group of people."},
    {"family": "universal",
     "text": "Respect human dignity: treat every person as worthy in themselves, never merely as a means to an end."},
    {"family": "universal",
     "text": "Be honest. Deceiving someone to gain an advantage over them is wrong, even when the lie is small."},
    {"family": "universal",
     "text": "Protect the privacy of others. Personal information should not be exposed without consent."},
    {"family": "universal",
     "text": "Children and other vulnerable people deserve special protection. Exploiting their vulnerability is gravely wrong."},
    {"family": "universal",
     "text": "Discrimination on the basis of race, gender, religion, or other immutable characteristics is unjust."},
    {"family": "universal",
     "text": "Helping someone in need at low cost to yourself is usually the right thing to do."},
    {"family": "universal",
     "text": "Gratuitous cruelty — causing suffering for no benefit — is always wrong."},
    # Operational rules for the "is this action acceptable?" framing:
    {"family": "universal",
     "text": "Taking, damaging, or using someone else's property without their permission is normally unacceptable."},
    {"family": "universal",
     "text": "Everyday acts that harm no one and violate no one's rights — ordinary chores, work, hobbies, caring for family — are morally acceptable."},
    {"family": "universal",
     "text": "Acts that deceive, steal from, endanger, or betray another person are normally unacceptable, even if the speaker felt they had a reason."},
    {"family": "universal",
     "text": "Keeping a commitment, caring for dependents, and meeting one's everyday responsibilities are acceptable and usually praiseworthy."},
    {"family": "universal",
     "text": "Breaking the law or a clear social rule purely for personal gain or convenience is normally unacceptable."},

    # ─── JUSTICE ───────────────────────────────────────────────────────────
    # ETHICS/justice asks whether a stated claim about desert or fairness is
    # *reasonable*. The discriminating question is whether the cited ground is
    # relevant and applied consistently.
    {"family": "justice",
     "text": "Treat similar cases similarly. If two people have done the same thing in the same circumstances, they should be treated the same way."},
    {"family": "justice",
     "text": "A claim of desert is reasonable when the reason it cites (effort, contribution, need, prior agreement, merit) is actually relevant to what is being distributed."},
    {"family": "justice",
     "text": "A claim is unreasonable when it rests on an irrelevant feature (a person's race, looks, mood, or an unrelated favour) or treats like cases differently for no relevant reason."},
    {"family": "justice",
     "text": "People who work harder, take greater risks, or contribute more often have a stronger claim to a larger share — but only when those criteria are relevant to the matter at hand."},
    {"family": "justice",
     "text": "Punishment should be proportional to the offence. Disproportionate penalties are themselves unjust."},
    {"family": "justice",
     "text": "A person should not be punished for a crime they did not commit, regardless of how convenient that would be."},
    {"family": "justice",
     "text": "Need can be a legitimate ground for desert: a starving person's claim to food is stronger than that of someone already well fed."},
    {"family": "justice",
     "text": "Past harm creates obligations. Someone who has wronged another typically owes them restitution or repair."},
    {"family": "justice",
     "text": "Privileges within a role come with corresponding responsibilities; one cannot claim the benefits without accepting the duties."},
    {"family": "justice",
     "text": "Expecting different treatment for the same conduct, with no relevant difference between the cases, is an unreasonable claim."},

    # ─── DEONTOLOGY ────────────────────────────────────────────────────────
    # ETHICS/deontology asks whether an *excuse* for declining a request, or a
    # claimed *role duty*, is reasonable. The test is the source of the reason.
    {"family": "deontology",
     "text": "An excuse for not doing a requested task is reasonable only when it cites a genuine conflicting obligation or a real inability, not mere preference, laziness, or convenience."},
    {"family": "deontology",
     "text": "Excuses that appeal to a prior commitment, a competing duty, a real incapacity, or someone else's stronger need are reasonable; excuses that amount to 'I don't feel like it' or 'it benefits me' are not."},
    {"family": "deontology",
     "text": "A duty that genuinely follows from one's role (a teacher teaching, a parent caring for a child, an employee doing their assigned job) is a reasonable expectation; a duty someone invents to claim a privilege they are not owed is not."},
    {"family": "deontology",
     "text": "Keep your promises. Breaking a promise is wrong unless a stronger moral duty overrides it."},
    {"family": "deontology",
     "text": "Doing something to a person without their informed consent violates their autonomy, even when you expect a good outcome."},
    {"family": "deontology",
     "text": "Lying is wrong even when telling the truth would be inconvenient. The duty of honesty does not bend to comfort."},
    {"family": "deontology",
     "text": "You have stronger duties to those who depend on you (children, patients, students) than to strangers, though you still have duties to strangers."},
    {"family": "deontology",
     "text": "An excuse for declining or breaking a duty is reasonable only when it appeals to a competing moral consideration of comparable weight, not to mere convenience."},
    {"family": "deontology",
     "text": "Using someone as a tool for someone else's benefit, without regard for their own interests, is wrong."},
    {"family": "deontology",
     "text": "If you take on a role (parent, employee, friend), you implicitly accept its duties; later inconvenience does not excuse you from them."},

    # ─── VIRTUE ────────────────────────────────────────────────────────────
    # ETHICS/virtue asks whether a scenario exemplifies a NAMED trait. The
    # model needs trait definitions and the actions that signal each trait.
    {"family": "virtue",
     "text": "A scenario exhibits a trait only when the character's action is a clear instance of that specific trait; a related-sounding but different trait does not count."},
    {"family": "virtue",
     "text": "Courage/bravery: facing danger, fear, or social pressure to do the right thing. Cowardice is its opposite."},
    {"family": "virtue",
     "text": "Honesty/truthfulness: telling the truth even when costly. Deceitfulness and dishonesty are opposites."},
    {"family": "virtue",
     "text": "Compassion/kindness: noticing another's suffering and acting to relieve it. Cruelty and callousness are opposites."},
    {"family": "virtue",
     "text": "Humility/modesty: acknowledging one's limits and accepting correction. Arrogance is its opposite."},
    {"family": "virtue",
     "text": "Generosity: freely sharing time, money, or attention. Greed and stinginess are opposites."},
    {"family": "virtue",
     "text": "Patience: staying calm and composed while waiting or under provocation. Irritability and impatience are opposites."},
    {"family": "virtue",
     "text": "Loyalty: standing by people or commitments under pressure. Betrayal is its opposite."},
    {"family": "virtue",
     "text": "A trait label fits only if the described behaviour matches that trait's definition; if the behaviour shows a different or opposite quality, the trait is not exhibited."},
    {"family": "virtue",
     "text": "Integrity means aligning one's actions with one's stated values, especially when the two are in tension."},
    {"family": "virtue",
     "text": "Common vices include cowardice, dishonesty, cruelty, arrogance, greed, callousness, and impatience."},

    # ─── UTILITY ───────────────────────────────────────────────────────────
    # ETHICS/utilitarianism asks which of two experiences is more PLEASANT.
    # Hedonic heuristics that rank experiences are what help here.
    {"family": "utility",
     "text": "An experience is more pleasant when it brings comfort, enjoyment, connection, success, or relief, and less pleasant when it brings pain, fear, loss, embarrassment, frustration, or boredom."},
    {"family": "utility",
     "text": "Getting what one wanted (a good meal, good news, kindness, a reward, a problem solved) is pleasant; being denied, hurt, insulted, delayed, or burdened is unpleasant."},
    {"family": "utility",
     "text": "All else equal, an action that causes more pleasure and less pain across the people it affects is morally preferable."},
    {"family": "utility",
     "text": "Long-term consequences matter as much as immediate ones. An action that brings brief relief but lasting harm is usually a bad trade."},
    {"family": "utility",
     "text": "Activities like eating well, talking with friends, learning, and resting tend to be pleasant; injury, exhaustion, isolation, and hunger tend to be unpleasant."},
    {"family": "utility",
     "text": "Routine errands are usually mildly pleasant or neutral. Activities that cause pain, embarrassment, or stress are unpleasant. The most pleasant moments often involve close relationships, novelty, or relief from suffering."},
    {"family": "utility",
     "text": "A pleasant version of an event becomes less pleasant when something goes wrong, is taken away, or is spoiled; the same event is more pleasant when it goes smoothly or exceeds expectations."},
    {"family": "utility",
     "text": "Free time spent on something one enjoys is generally more pleasant than the same time spent on a chore or obligation."},
    {"family": "utility",
     "text": "Aggregating wellbeing across people requires care: a small loss to many can outweigh a large gain to one, and vice versa."},
]


# ════════════════════════════════════════════════════════════════════════════
#  RagWrapper
# ════════════════════════════════════════════════════════════════════════════

class RagWrapper:
    """
    Embedder + FAISS index + corpus. Public API:

        rag.retrieve(query)               → list[dict]
        rag.format_guidelines(query, k)   → list[str]
    """

    def __init__(self, embedder, index, corpus: List[dict]):
        self.embedder = embedder
        self.index = index
        self.corpus = corpus

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def build(cls, *, save_to: str = C.RAG_INDEX_OUT) -> "RagWrapper":
        from sentence_transformers import SentenceTransformer
        import faiss

        print(f"[rag] loading embedder   {C.EMBED_MODEL}")
        embedder = SentenceTransformer(C.EMBED_MODEL)

        print(f"[rag] encoding corpus    ({len(CORPUS)} docs)")
        texts = [d["text"] for d in CORPUS]
        embs = embedder.encode(texts, convert_to_numpy=True,
                               show_progress_bar=False)
        embs = embs.astype(np.float32)
        faiss.normalize_L2(embs)

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        os.makedirs(save_to, exist_ok=True)
        faiss.write_index(index, os.path.join(save_to, "faiss.index"))
        with open(os.path.join(save_to, "corpus.json"), "w") as f:
            json.dump(CORPUS, f, indent=2, ensure_ascii=False)

        print(f"[rag] index written      {save_to}")
        return cls(embedder, index, CORPUS)

    @classmethod
    def from_disk(cls, path: str = C.RAG_INDEX_OUT) -> "RagWrapper":
        from sentence_transformers import SentenceTransformer
        import faiss

        idx_path = os.path.join(path, "faiss.index")
        corpus_path = os.path.join(path, "corpus.json")
        if not os.path.exists(idx_path):
            print(f"[rag] no index at {path}; building one…")
            return cls.build(save_to=path)

        # Guard against a stale index built from an older/smaller corpus:
        # if the on-disk corpus no longer matches the in-code CORPUS, rebuild.
        try:
            with open(corpus_path) as f:
                disk_corpus = json.load(f)
        except FileNotFoundError:
            disk_corpus = None
        if not disk_corpus or len(disk_corpus) != len(CORPUS):
            print(f"[rag] on-disk corpus is stale "
                  f"({0 if not disk_corpus else len(disk_corpus)} docs vs "
                  f"{len(CORPUS)} in code); rebuilding index…")
            return cls.build(save_to=path)

        embedder = SentenceTransformer(C.EMBED_MODEL)
        index = faiss.read_index(idx_path)
        return cls(embedder, index, disk_corpus)

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = C.RAG_TOP_K,
                 *, families=None, exclude_families=None) -> List[dict]:
        """
        Return the top-`top_k` most relevant principles for `query`.

        families         : if given, keep only principles whose family is in
                           this set (retrieval *routing*). This is the key
                           change vs. the original un-routed search.
        exclude_families : if given, drop principles in these families
                           (used by the family-removal ablation).

        Because filtering happens after the FAISS search, we over-retrieve
        (search the whole index) and then keep the first `top_k` survivors in
        similarity order.
        """
        import faiss
        q = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q)

        # Over-retrieve so that, after family filtering, we still have enough.
        n_search = len(self.corpus)
        _, idxs = self.index.search(q, n_search)

        fam_keep = set(families) if families else None
        fam_drop = set(exclude_families) if exclude_families else set()

        out: List[dict] = []
        for i in idxs[0]:
            if not (0 <= i < len(self.corpus)):
                continue
            d = self.corpus[i]
            fam = d.get("family")
            if fam in fam_drop:
                continue
            if fam_keep is not None and fam not in fam_keep:
                continue
            out.append(d)
            if len(out) >= top_k:
                break
        return out

    def format_guidelines(self, query: str, top_k: int = C.RAG_TOP_K,
                          *, families=None, exclude_families=None) -> List[str]:
        return [d["text"] for d in self.retrieve(
            query, top_k=top_k,
            families=families, exclude_families=exclude_families)]


# ════════════════════════════════════════════════════════════════════════════
#  Sanity-check CLI
# ════════════════════════════════════════════════════════════════════════════

def _demo():
    rag = RagWrapper.from_disk()
    queries = [
        "Is it okay to lie to spare someone's feelings?",
        "Should I keep a promise to a friend if it would hurt someone else?",
        "Eating a delicious meal at a restaurant.",
        "A judge knowingly convicts an innocent man to deter future crime.",
        "Sarah notices her friend dropped a twenty and quietly pockets it.",
    ]
    for q in queries:
        print(f"\nQ  {q}")
        for d in rag.retrieve(q):
            print(f"   [{d['family']:>10}]  {d['text']}")


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_index", action="store_true",
                    help="(re-)build the FAISS index from the in-code corpus")
    ap.add_argument("--demo", action="store_true",
                    help="show retrieval output on a few sample queries")
    args = ap.parse_args()

    if args.build_index:
        RagWrapper.build()
    if args.demo:
        _demo()


if __name__ == "__main__":
    _cli()
