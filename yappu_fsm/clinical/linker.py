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

    PRE_TRIGGERS: Dict[str, List[str]] = {
        "NEGATED": [r"\bdenies\b", r"\bdenied\b", r"\bno evidence of\b", r"\bnegative for\b",
                    r"\bwithout\b", r"\bno\b", r"\bfree of\b"],
        "UNCERTAIN": [r"\brule out\b", r"\br/o\b", r"\bpossible\b", r"\bsuspected\b",
                      r"\bprobable\b", r"\bquestionable\b", r"\bconcern for\b"],
        "HISTORICAL": [r"\bhistory of\b", r"\bh/o\b", r"\bpmh\b", r"\bstatus post\b", r"\bs/p\b",
                       r"\bprevious\b", r"\bprior\b"],
        "FAMILY_HISTORY": [r"\bfamily history of\b", r"\bfhx\b", r"\bmother had\b", r"\bfather had\b",
                           r"\bmother has\b", r"\bfather has\b", r"\bsister had\b", r"\bbrother had\b"],
    }
    POST_TRIGGERS: Dict[str, List[str]] = {
        "NEGATED": [r"^\W*(?:was |is |has been )?ruled out\b", r"^\W*(?:is )?not present\b"],
        "HISTORICAL": [r"^\W*(?:has |is |was )?resolved\b"],
        "UNCERTAIN": [r"^\W*(?:is )?(?:suspected|unlikely|possible)\b"],
    }
    PSEUDO_TRIGGERS = [r"\bno change\b", r"\bno increase\b", r"\bnot only\b", r"\bgram negative\b",
                       r"\bwithout difficulty\b", r"\bnot ruled out\b"]
    # "family history of" must win over "history of": checked by longest-match below.

    WALLS = [r"\bbut\b", r"\bhowever\b", r"\bexcept\b", r"\balthough\b", r"\bwhereas\b",
             r"\.(?!\d)", r";", r"\n"]

    def __init__(self, lexicon: Lexicon, max_laterality_distance: int = 4,
                 fuzzy_threshold: float = 0.84, min_fuzzy_word_len: int = 5,
                 post_window_chars: int = 30, max_scope_tokens: int = 6,
                 setting: str = "outpatient"):
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
                for e in self.index:
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
                    if any(a <= m.start() < b for a, b in pseudo):
                        continue
                    hits.append((m.start(), m.end(), state))
        # keep the longest of overlapping triggers ("family history of" beats "history of")
        kept: List[Tuple[int, int, str]] = []
        for h in sorted(hits, key=lambda h: -(h[1] - h[0])):
            if all(h[1] <= k[0] or h[0] >= k[1] for k in kept):
                kept.append(h)
        return sorted(kept)

    def _assertion(self, clause_l: str, triggers, c0: int, c1: int) -> str:
        state = "PRESENT"
        pre = [t for t in triggers if t[1] <= c0
               and len(re.findall(r"[\w/'-]+", clause_l[t[1]:c0])) <= self.max_scope]
        if pre:
            state = max(pre, key=lambda t: t[1])[2]
        tail = clause_l[c1:c1 + self.post_window]
        for st, pats in self.POST_TRIGGERS.items():
            if any(re.search(p, tail) for p in pats):
                if not any(re.search(p, tail) for p in self.PSEUDO_TRIGGERS):
                    state = st
        return state

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
                lat, dist = self._bind_laterality(ent["t0"], ent["t1"], lats)
                cmap = ent["entry"]["code_map"]
                lookup = assertion
                if assertion == "UNCERTAIN" and self.setting == "inpatient":
                    lookup = "PRESENT"   # CM Section II.H: code as if established
                if lookup in ("NEGATED", "UNCERTAIN"):
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
