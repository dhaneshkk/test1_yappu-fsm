"""Rule-based clinical entity recognition and ICD-10 linking.

Pipeline for each note:

1. expand abbreviations (case-sensitive for ambiguous ones such as CAP, OD, OS, MI);
2. split into clauses at boundary "walls" (., ;, newline, but, however, ...) –
   the Sandhi analogy: a wall stops assertion scope from leaking;
3. find lexicon entities by exact or fuzzy (phonetic / edit-distance) match;
4. assign each entity the assertion of the **nearest preceding** trigger in its
   clause (NEGATED / UNCERTAIN / HISTORICAL / FAMILY_HISTORY), or a post-trigger
   right after it; pseudo-triggers ("no change", "gram negative") are ignored;
5. bind laterality (RIGHT / LEFT / BILATERAL) by closest-token distance;
6. look the (assertion, laterality) key up in the lexicon to get code(s).

Coding setting follows the ICD-10-CM Official Guidelines for Coding and Reporting:

* ``setting="outpatient"`` (default, Section IV.H): uncertain diagnoses ("probable",
  "suspected", "rule out", ...) are NOT coded; code the documented signs/symptoms instead.
* ``setting="inpatient"`` (Section II.H / III.C): an uncertain diagnosis documented at
  discharge is coded as if it existed or was established.

Negated findings are never coded. This is decision support only: output must be
reviewed by a qualified coder.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .phonetic import clinical_phonetic_key, phonetic_word, similarity

CodeValue = Union[str, List[str]]
Lexicon = Dict[str, Dict[str, CodeValue]]

SETTINGS = ("outpatient", "inpatient")


@dataclass
class Finding:
    raw_mention: str
    entity: str
    match_type: str
    confidence: float
    assertion: str
    laterality: str
    laterality_distance: Optional[int]
    codes: List[str]
    status: str               # CODE | DO_NOT_CODE | REVIEW_SPECIFICITY
    clause: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_lexicon(path: Union[str, Path]) -> Lexicon:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


class ClinicalLinker:
    # (pattern, replacement, case_sensitive)
    ABBREVIATIONS: List[Tuple[str, str, bool]] = [
        (r"\bt2dm\b", "type 2 diabetes", False),
        (r"\bhtn\b", "hypertension", False),
        (r"\bsob\b", "shortness of breath", False),
        (r"\bnpdr\b", "nonproliferative diabetic retinopathy", False),
        (r"\bb/l\b", "bilateral", False),
        (r"\ba-?fib\b", "atrial fibrillation", False),
        (r"\bchf\b", "heart failure", False),
        (r"\bckd\b", "chronic kidney disease", False),
        (r"\buti\b", "urinary tract infection", False),
        (r"\bhld\b", "hyperlipidemia", False),
        (r"\bcva\b", "stroke", False),
        (r"\bcts\b", "carpal tunnel syndrome", False),
        (r"\bt1dm\b", "type 1 diabetes", False),
        (r"\brt\b", "right", False),
        (r"\blt\b", "left", False),
        (r"\bMI\b", "myocardial infarction", True),
        (r"\bCAP\b", "pneumonia", True),
        (r"\bOD\b", "right eye", True),
        (r"\bOS\b", "left eye", True),
        (r"\bOU\b", "bilateral eyes", True),
    ]

    LATERALITY: Dict[str, List[str]] = {
        "BILATERAL": [r"\bbilateral eyes\b", r"\bboth eyes\b", r"\bbilateral\b", r"\bboth\b"],
        "RIGHT": [r"\bright eye\b", r"\bright-sided\b", r"\bright\b"],
        "LEFT": [r"\bleft eye\b", r"\bleft-sided\b", r"\bleft\b"],
    }

    # Trigger lists follow NegEx (Chapman et al. 2001) and its ConText extension
    # (Harkema et al. 2009): pre-triggers scan forward, post-triggers scan backward, both
    # over at most ``max_scope_tokens`` tokens and never across a CONJ terminator.
    PRE_TRIGGERS: Dict[str, List[str]] = {
        "NEGATED": [r"\bdenies\b", r"\bdenied\b", r"\bno evidence of\b", r"\bnegative for\b",
                    r"\bwithout\b", r"\bno\b", r"\bfree of\b", r"\bruled out\b", r"\brules out\b",
                    r"\babsence of\b", r"\bno signs? of\b", r"\bno history of\b", r"\bnot\b",
                    r"\bnever had\b", r"\bnone\b"],
        # Outpatient uncertain-diagnosis terms (ICD-10-CM Section IV.H) incl. "consistent with" / "compatible with"
        "UNCERTAIN": [r"\brule out\b", r"\br/o\b", r"\bpossible\b", r"\bsuspected\b",
                      r"\bprobable\b", r"\bquestionable\b", r"\bconcern for\b", r"\bconcerning for\b",
                      r"\bsuspicious for\b", r"\bruling out\b", r"\bconsistent with\b", r"\bcompatible with\b",
                      r"\bpending confirmation of\b", r"\bworking diagnosis(?: of)?\b", r"\bversus\b", r"\bvs\.?(?=\s)",
                      r"\bnot (?:been )?ruled out\b", r"\bcannot be ruled out\b", r"\bnot rule out\b"],
        # "History of present illness" is a note section header, not a history statement
        "HISTORICAL": [r"\bhistory of\b(?! present illness)", r"\bh/o\b", r"\bpmh\b", r"\bstatus post\b", r"\bs/p\b",
                       r"\bprevious\b", r"\bprior\b"],
        "FAMILY_HISTORY": [r"\bfamily history of\b", r"\bfhx\b", r"\bmother had\b", r"\bfather had\b",
                           r"\bmother has\b", r"\bfather has\b", r"\bsister had\b", r"\bbrother had\b"],
        "HYPOTHETICAL": [r"\bif (?:the patient |he |she |you |they )?develops?\b", r"\breturn if\b",
                         r"\bcall if\b", r"\bshould (?:the patient |he |she |you |they )?develop\b",
                         r"\bin case of\b", r"\bwatch for\b"],
    }
    POST_TRIGGERS: Dict[str, List[str]] = {
        "NEGATED": [r"\b(?:was |is |has been |were )?ruled out\b", r"\b(?:is |was )?not present\b",
                    r"\bnot demonstrated\b", r"\bnot seen\b", r"\bnever developed\b", r"\babsent\b"],
        "HISTORICAL": [r"\b(?:has |is |was )?resolved\b"],
        "UNCERTAIN": [r"\b(?:is |was )?(?:suspected|unlikely|likely|possible|questionable)\b",
                      r"\bcannot be (?:ruled out|excluded)\b", r"\bnot (?:been )?ruled out\b",
                      r"\bnot yet confirmed\b", r"\bpending confirmation\b"],
    }
    # A negation that attaches to a feature of the diagnosis, not the diagnosis itself:
    # "no COPD exacerbations", "denies asthma attacks", "no gout flares" → the diagnosis stays PRESENT.
    # ("admission"/"hospitalization" are deliberately absent: "ruled out pneumonia on admission" is a timing phrase)
    FEATURE_NOUNS = r"(?:exacerbations?|flares?|flare-ups?|episodes?|attacks?|symptoms?|complications?|crises|crisis)"
    # Pseudo-triggers stop a NEGATED trigger from opening a scope (NegEx [PSEU]).
    PSEUDO_TRIGGERS = [r"\bno change\b", r"\bno increase\b", r"\bnot only\b", r"\bgram negative\b",
                       r"\bwithout difficulty\b", r"\bnot (?:been )?ruled out\b", r"\bnot rule out\b",
                       r"\bcannot be ruled out\b", r"\bno further\b", r"\bnot certain whether\b"]
    # CONJ terminators end a scope inside a clause (NegEx [CONJ]).
    CONJ = [r"\baside from\b", r"\bapart from\b", r"\bsecondary to\b", r"\bcause of\b", r"\bsource of\b",
            r"\bdue to\b", r"\bexcept\b"]
    # Clause walls (also CONJ terminators) split the note before any matching.
    WALLS = [r"\bbut\b", r"\bhowever\b", r"\bexcept\b", r"\balthough\b", r"\bwhereas\b",
             r"\.(?!\d)", r";", r"\n"]

    def __init__(self, lexicon: Lexicon, max_laterality_distance: int = 4,
                 fuzzy_threshold: float = 0.84, min_fuzzy_word_len: int = 5,
                 post_window_chars: int = 30, max_scope_tokens: int = 6,
                 setting: str = "outpatient", index: bool = True):
        if setting not in SETTINGS:
            raise ValueError(f"setting must be one of {SETTINGS}")
        self.setting = setting
        self.lexicon = {k.lower(): v for k, v in lexicon.items()}
        self.max_lat = max_laterality_distance
        self.fuzzy_threshold = fuzzy_threshold
        self.min_fuzzy_len = min_fuzzy_word_len
        self.post_window = post_window_chars
        self.max_scope = max_scope_tokens
        self.index = [{"canonical": k, "words": k.split(), "code_map": v} for k, v in self.lexicon.items()]
        self.max_ngram = max((len(e["words"]) for e in self.index), default=1)
        self.use_index = index
        self._first_word_index = defaultdict(set)
        self._cand_cache: Dict[str, Tuple[int, ...]] = {}
        if index:
            for i, e in enumerate(self.index):
                for key in self._index_keys(e["words"][0]):
                    self._first_word_index[key].add(i)

    # ------------------------------------------------------------------ index
    def _index_keys(self, w: str):
        """Hash keys for candidate lookup: exact word, phonetic skeleton and
        SymSpell-style one-deletion variants (symmetric, so they cover a single
        substitution, insertion, deletion or adjacent transposition)."""
        yield "x:" + w
        if len(w) >= self.min_fuzzy_len:
            yield "p:" + phonetic_word(w)
            yield "d:" + w
            for i in range(len(w)):
                yield "d:" + w[:i] + w[i + 1:]

    def _candidates(self, first_word: str) -> Sequence[int]:
        """Indices of lexicon entries whose first word could match ``first_word``."""
        if not self.use_index:
            return range(len(self.index))
        hit = self._cand_cache.get(first_word)
        if hit is None:
            found = set()
            for key in self._index_keys(first_word):
                found |= self._first_word_index.get(key, set())
            hit = tuple(sorted(found))
            if len(self._cand_cache) < 100_000:
                self._cand_cache[first_word] = hit
        return hit

    # ------------------------------------------------------------------ text
    def expand_abbreviations(self, text: str) -> str:
        for pat, rep, cs in self.ABBREVIATIONS:
            text = re.sub(pat, rep, text, flags=0 if cs else re.IGNORECASE)
        return text

    def split_clauses(self, text: str) -> List[str]:
        parts = re.split("|".join(self.WALLS), text, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def _tokens(text: str) -> List[Tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w/'-]+", text)]

    @staticmethod
    def _char_to_tok(c0: int, c1: int, toks) -> Tuple[int, int]:
        s = next((i for i, t in enumerate(toks) if t[2] > c0), len(toks) - 1)
        e = max((i for i, t in enumerate(toks) if t[1] < c1), default=s)
        return s, e

    # --------------------------------------------------------------- matching
    def _word_match(self, a: str, b: str) -> Optional[str]:
        if a == b:
            return "EXACT"
        if len(a) < self.min_fuzzy_len or len(b) < self.min_fuzzy_len:
            return None
        if phonetic_word(a) == phonetic_word(b) and similarity(a, b) >= 0.75:
            return "PHONETIC"
        if similarity(a, b) >= self.fuzzy_threshold:
            return "EDIT"
        return None

    def match_entities(self, toks) -> List[dict]:
        words = [t[0].lower() for t in toks]
        taken, found = set(), []
        for n in range(self.max_ngram, 0, -1):
            for i in range(len(words) - n + 1):
                if any(j in taken for j in range(i, i + n)):
                    continue
                span = words[i:i + n]
                best = None
                for ci in self._candidates(words[i]):
                    e = self.index[ci]
                    if len(e["words"]) != n:
                        continue
                    kinds = [self._word_match(a, b) for a, b in zip(span, e["words"])]
                    if None in kinds:
                        continue
                    phrase, canon = " ".join(span), e["canonical"]
                    if all(k == "EXACT" for k in kinds):
                        mt, score = "EXACT", 1.0
                    else:
                        mt = "PHONETIC_FUZZY" if "EDIT" not in kinds else "EDIT_FUZZY"
                        score = similarity(phrase, canon)
                    if best is None or score > best[1]:
                        best = (e, score, mt)
                if best:
                    e, score, mt = best
                    taken.update(range(i, i + n))
                    found.append({"raw": " ".join(t[0] for t in toks[i:i + n]), "entry": e,
                                  "score": round(score, 3), "match_type": mt,
                                  "c0": toks[i][1], "c1": toks[i + n - 1][2], "t0": i, "t1": i + n - 1})
        return sorted(found, key=lambda x: x["t0"])

    # -------------------------------------------------------------- assertion
    def _triggers(self, clause_l: str) -> List[Tuple[int, int, str]]:
        pseudo = [(m.start(), m.end()) for p in self.PSEUDO_TRIGGERS for m in re.finditer(p, clause_l)]
        hits = []
        for state, pats in self.PRE_TRIGGERS.items():
            for p in pats:
                for m in re.finditer(p, clause_l):
                    if state == "NEGATED" and any(a <= m.start() < b for a, b in pseudo):
                        continue
                    hits.append((m.start(), m.end(), state))
        # keep the longest of overlapping triggers ("family history of" beats "history of")
        kept: List[Tuple[int, int, str]] = []
        for h in sorted(hits, key=lambda h: -(h[1] - h[0])):
            if all(h[1] <= k[0] or h[0] >= k[1] for k in kept):
                kept.append(h)
        return sorted(kept)

    def _ntok(self, text: str) -> int:
        return len(re.findall(r"[\w/'-]+", text))

    def _conj_between(self, clause_l: str, a: int, b: int) -> bool:
        seg = clause_l[a:b]
        return any(re.search(p, seg) for p in self.CONJ)

    def _post_hits(self, clause_l: str) -> List[Tuple[int, int, str]]:
        pseudo = [(m.start(), m.end()) for p in self.PSEUDO_TRIGGERS for m in re.finditer(p, clause_l)]
        hits = []
        for state, pats in self.POST_TRIGGERS.items():
            for p in pats:
                for m in re.finditer(p, clause_l):
                    if state == "NEGATED" and any(a <= m.start() < b for a, b in pseudo):
                        continue
                    hits.append((m.start(), m.end(), state))
        return hits

    def _assertion(self, clause_l: str, triggers, c0: int, c1: int) -> str:
        """NegEx/ConText scope: the nearest pre-trigger before the entity, within the token
        window and with no CONJ terminator in between; a post-trigger after the entity under
        the same conditions overrides it."""
        state = "PRESENT"
        pre = [t for t in triggers if t[1] <= c0 and self._ntok(clause_l[t[1]:c0]) <= self.max_scope
               and not self._conj_between(clause_l, t[1], c0)]
        if pre:
            state = max(pre, key=lambda t: t[1])[2]
        post = [t for t in self._post_hits(clause_l) if t[0] >= c1 and self._ntok(clause_l[c1:t[0]]) <= self.max_scope
                and not self._conj_between(clause_l, c1, t[0])]
        if post:
            state = min(post, key=lambda t: t[0])[2]
        return state

    def _feature_negation(self, clause_l: str, c1: int) -> bool:
        """True if a feature noun directly follows the entity (at most one word in between),
        so a pre-trigger such as "no" negates the feature, not the diagnosis."""
        return re.match(r"^\s+(?:[\w-]+\s+)?" + self.FEATURE_NOUNS + r"\b", clause_l[c1:]) is not None

    # ------------------------------------------------------------- laterality
    def _laterality_mentions(self, clause_l: str, toks) -> List[dict]:
        out = []
        for label, pats in self.LATERALITY.items():
            for p in pats:
                for m in re.finditer(p, clause_l):
                    t0, t1 = self._char_to_tok(m.start(), m.end(), toks)
                    if any(not (t1 < o["t0"] or t0 > o["t1"]) for o in out):
                        continue
                    out.append({"label": label, "t0": t0, "t1": t1, "used": False})
        return out

    def _bind_laterality(self, t0: int, t1: int, mentions) -> Tuple[str, Optional[int]]:
        best, best_d = None, None
        for m in mentions:
            if m["used"]:
                continue
            d = t0 - m["t1"] if m["t1"] < t0 else (m["t0"] - t1 if m["t0"] > t1 else 0)
            if d <= self.max_lat and (best_d is None or d < best_d):
                best, best_d = m, d
        if best is None:
            return "UNSPECIFIED", None
        best["used"] = True
        return best["label"], best_d

    # ---------------------------------------------------------------- public
    def process(self, note: str) -> List[Finding]:
        findings: List[Finding] = []
        for clause in self.split_clauses(self.expand_abbreviations(note)):
            cl = clause.lower()
            toks = self._tokens(cl)
            if not toks:
                continue
            triggers = self._triggers(cl)
            lats = self._laterality_mentions(cl, toks)
            for ent in self.match_entities(toks):
                assertion = self._assertion(cl, triggers, ent["c0"], ent["c1"])
                if assertion == "NEGATED" and self._feature_negation(cl, ent["c1"]):
                    assertion = "PRESENT"
                lat, dist = self._bind_laterality(ent["t0"], ent["t1"], lats)
                cmap = ent["entry"]["code_map"]
                lookup = assertion
                if assertion == "UNCERTAIN" and self.setting == "inpatient":
                    lookup = "PRESENT"   # CM Section II.H: code as if established
                if lookup in ("NEGATED", "UNCERTAIN", "HYPOTHETICAL"):
                    codes, status = [], "DO_NOT_CODE"
                else:
                    val = cmap.get(f"{lookup}_{lat}") or cmap.get(f"{lookup}_UNSPECIFIED") or cmap.get(lookup)
                    codes = [] if val is None else ([val] if isinstance(val, str) else list(val))
                    status = "CODE" if codes else "REVIEW_SPECIFICITY"
                findings.append(Finding(ent["raw"], ent["entry"]["canonical"], ent["match_type"], ent["score"],
                                        assertion, lat, dist, codes, status, clause))
        return findings

    @staticmethod
    def allowed_codes(findings: Sequence[Finding]) -> List[str]:
        """Codes the ICD-10 constrained decoder may emit for this note."""
        out: List[str] = []
        for f in findings:
            for c in f.codes:
                if c not in out:
                    out.append(c)
        return out
