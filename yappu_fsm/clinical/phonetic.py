"""Phonetic skeleton keys and fuzzy similarity for clinical terms.

The skeleton plays the role of Monai's Ina Ezhuthu equivalence classes:
spelling variants that sound alike (ph/f, pn-/n-, c/k/q, z/s, doubled letters,
dropped interior vowels) collapse to the same key.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def phonetic_word(w: str) -> str:
    w = w.lower()
    if len(w) <= 3:
        return w.upper()
    w = re.sub(r"^pn", "n", w)
    w = re.sub(r"^kn", "n", w)
    w = re.sub(r"^ps", "s", w)
    w = w.replace("ph", "f").replace("gh", "g").replace("ck", "k")
    w = re.sub(r"[cq]", "k", w)
    w = w.replace("z", "s").replace("y", "i")
    w = re.sub(r"([a-z])\1+", r"\1", w)
    return w[0].upper() + re.sub(r"[aeiou]", "", w[1:]).upper()


def clinical_phonetic_key(phrase: str) -> str:
    """'osteoarthritis' and 'osteoarthritus' -> 'OSTRTHRTS'."""
    return " ".join(phonetic_word(w) for w in re.findall(r"[a-zA-Z]+", phrase))


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
