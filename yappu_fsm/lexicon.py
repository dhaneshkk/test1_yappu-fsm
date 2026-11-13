"""Annotated word lexicon used by the constrained Venba decoder."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from .prosody import EETRUCHEER, LINE_STARTS, VENBA_CATEGORIES, classify, eetrucheer_vaipadu, expected_next_start
from .tamil import encode_asai, ethukai_key, monai_key, normalize


@dataclass(frozen=True)
class Entry:
    word: str
    asai: str
    category: Optional[str]
    final_ok: bool          # usable as Eetrucheer
    ethukai: Optional[str]
    monai: Optional[str]


# Words from the three poems in this project plus a handful of extras.
SAMPLE_WORDS = """
எண்ணும் எழுத்தும் இசையோடு கூடினால் பண்ணும் மணியாகும் பார்ப்பவர் கண்ணும்
அமுதத் தமிழில் அமைந்த இலக்கணம் நம்மொழி யாக்கும் இசை
வண்ண நிலவொளி வானில் தவழ்ந்திட எண்ணற்ற விண்மீன்கள் ஏற்றிடும் விண்ணில்
இருளினை நீக்கும் இனிய ஒளியால் மருளினை மாற்றும் மதி
அடக்க முடையா ரறிவிலரென் றெண்ணிக் கடக்கக் கருதவும் வேண்டா மடைத்தலையில்
ஓடுமீ னோட உறுமீன் வருமளவும் வாடி யிருக்குமாம் கொக்கு
மண்ணில் கண்ணில் பொருளினை அருளினை மலர் நாள் காசு பிறப்பு அன்பு
""".split()


class VenbaLexicon:
    """A set of words annotated with Asai, Seer category, Ethukai and Monai keys."""

    def __init__(self, words: Iterable[str], strict_ethukai: bool = False, scheme: str = "iru"):
        if scheme not in ("iru", "oru"):
            raise ValueError("scheme must be 'iru' or 'oru'")
        self.strict_ethukai = strict_ethukai
        self.scheme = scheme
        seen, entries = set(), []
        for w in words:
            w = normalize(w.strip())
            if not w or w in seen:
                continue
            seen.add(w)
            asai = encode_asai(w)
            if not asai:
                continue
            entries.append(Entry(w, asai, classify(asai)[1], eetrucheer_vaipadu(w) is not None,
                                 ethukai_key(w, strict_ethukai), monai_key(w)))
        self.entries: List[Entry] = entries
        self.by_word: Dict[str, Entry] = {e.word: e for e in entries}
        self._viable = lru_cache(maxsize=None)(self._viable_uncached)

    @classmethod
    def sample(cls, **kw) -> "VenbaLexicon":
        return cls(SAMPLE_WORDS, **kw)

    # ---- constraint logic -------------------------------------------------

    def _passes(self, e: Entry, k: int, expected: Optional[str], key_a: Optional[str], key_b: Optional[str]) -> bool:
        if k == EETRUCHEER:
            if not e.final_ok:
                return False
            # Kaasu/Pirappu start like Naal/Malar
        elif e.category not in VENBA_CATEGORIES:
            return False
        if expected is not None and not e.asai.startswith(expected):
            return False
        if k in (4, 7) and e.ethukai != key_a:
            return False
        if k == 8 and self.scheme == "oru" and e.ethukai != key_a:
            return False
        if k == 12 and e.ethukai != key_b:
            return False
        if k in (0, 8) and e.ethukai is None:
            return False
        return True

    def _next_keys(self, k: int, e: Entry, key_a, key_b):
        if k == 0:
            key_a = e.ethukai
        if k == 8:
            key_b = e.ethukai
        return expected_next_start(e.asai), key_a, key_b

    def _viable_uncached(self, k: int, expected, key_a, key_b) -> bool:
        """Is there at least one way to finish the poem from foot ``k``?"""
        for e in self.entries:
            if self._passes(e, k, expected, key_a, key_b):
                if k == EETRUCHEER:
                    return True
                if self._viable(k + 1, *self._next_keys(k, e, key_a, key_b)):
                    return True
        return False

    @lru_cache(maxsize=None)
    def candidates(self, k: int, expected, key_a, key_b) -> Tuple[Entry, ...]:
        """Words allowed at foot ``k`` that still leave a complete poem reachable."""
        out = []
        for e in self.entries:
            if not self._passes(e, k, expected, key_a, key_b):
                continue
            if k == EETRUCHEER or self._viable(k + 1, *self._next_keys(k, e, key_a, key_b)):
                out.append(e)
        return tuple(out)

    def line_start(self, k: int) -> int:
        return max(s for s in LINE_STARTS if s <= k)
