"""Seer classification, Thalai (linking) rules and a full Nerisai Venba validator."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .tamil import encode_asai, ethukai_key, has_kutriyalukaram_ending, split_display

# asai string -> (Vaipadu, category)
SEER: Dict[str, Tuple[str, str]] = {
    "11": ("Thema", "Maacheer"),
    "01": ("Pulima", "Maacheer"),
    "00": ("Karuvilam", "Vilacheer"),
    "10": ("Koovilam", "Vilacheer"),
    "111": ("Themangai", "Kaycheer"),
    "011": ("Pulimangai", "Kaycheer"),
    "001": ("Karuvilangai", "Kaycheer"),
    "101": ("Koovilangai", "Kaycheer"),
    "110": ("Themangani", "Kanicheer"),
    "010": ("Pulimangani", "Kanicheer"),
    "000": ("Karuvilangani", "Kanicheer"),
    "100": ("Koovilangani", "Kanicheer"),
}
VENBA_CATEGORIES = {"Maacheer", "Vilacheer", "Kaycheer"}

# Nerisai Venba layout: foot indices per line; foot 7 is the Thanichol, foot 14 the Eetrucheer.
LINE_FEET = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14)]
THANICHOL, EETRUCHEER = 7, 14
LINE_STARTS = (0, 4, 8, 12)


def classify(asai: str) -> Tuple[Optional[str], Optional[str]]:
    return SEER.get(asai, (None, None))


def eetrucheer_vaipadu(word: str) -> Optional[str]:
    """Return Naal / Malar / Kaasu / Pirappu if ``word`` is a valid final foot."""
    asai = encode_asai(word)
    if asai == "1":
        return "Naal"
    if asai == "0":
        return "Malar"
    if has_kutriyalukaram_ending(word):
        if asai == "11":
            return "Kaasu"
        if asai == "01":
            return "Pirappu"
    return None


def expected_next_start(prev_asai: str) -> Optional[str]:
    """Vendalai: after Maacheer the next foot starts with Nirai ('0'); after
    Vilacheer or Kaycheer it starts with Ner ('1')."""
    cat = classify(prev_asai)[1]
    if cat == "Maacheer":
        return "0"
    if cat in ("Vilacheer", "Kaycheer"):
        return "1"
    return None


def thalai_name(prev_asai: str, next_asai: str) -> str:
    cat = classify(prev_asai)[1]
    start = next_asai[:1]
    if cat == "Maacheer":
        return "Iyarseer Vendalai" if start == "0" else "Nerondru Aasiriyathalai"
    if cat == "Vilacheer":
        return "Iyarseer Vendalai" if start == "1" else "Niraiondru Aasiriyathalai"
    if cat == "Kaycheer":
        return "Venseer Vendalai" if start == "1" else "Kalithalai"
    if cat == "Kanicheer":
        return "Ondriya Vanjithalai" if start == "0" else "Ondraadha Vanjithalai"
    return "Unknown"


def is_vendalai(name: str) -> bool:
    return name in ("Iyarseer Vendalai", "Venseer Vendalai")


@dataclass
class FootReport:
    index: int
    word: str
    asai: str
    split: str
    vaipadu: Optional[str]
    category: Optional[str]
    thalai_in: Optional[str]
    ok: bool
    message: str = ""


@dataclass
class VenbaReport:
    valid: bool
    feet: List[FootReport] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    vikarpam: Optional[str] = None

    def lines(self) -> List[str]:
        out = []
        for f in self.feet:
            mark = "ok " if f.ok else "ERR"
            out.append(
                f"[{mark}] foot {f.index:02d} {f.word:<14} {f.split:<16} {f.asai:<4} "
                f"{(f.vaipadu or '-'):<13} {(f.thalai_in or '-'):<26} {f.message}"
            )
        out += [f"ERROR: {e}" for e in self.errors]
        out.append(f"Verdict: {'VALID' if self.valid else 'INVALID'} Nerisai Venba"
                   + (f" ({self.vikarpam})" if self.vikarpam else ""))
        return out


def parse_venba(text: str) -> List[str]:
    """Split poem text into words, dropping punctuation such as the Thanichol dash."""
    return [w for w in re.split(r"[\s\-–—.,;!?]+", text) if w]


def format_venba(words: Sequence[str]) -> str:
    """Lay 15 feet out as a 4-line Nerisai Venba with the Thanichol dash."""
    w = list(words)
    if len(w) != 15:
        return " ".join(w)
    return "\n".join([
        " ".join(w[0:4]),
        " ".join(w[4:7]) + " - " + w[7],
        " ".join(w[8:12]),
        " ".join(w[12:15]) + ".",
    ])


def validate_nerisai_venba(poem, strict_ethukai: bool = False) -> VenbaReport:
    """Validate a Nerisai Venba given as text or as a list of 15 words.

    Checks: 15 feet; only Maa/Vilam/Kay feet (no Kanicheer, no 4-asai feet);
    a valid Orasai/Kaasu/Pirappu Eetrucheer; Vendalai on all 14 links; and
    Ethukai (Oru or Iru Vikarpam) across lines 1-2 + Thanichol and lines 3-4.
    """
    words = parse_venba(poem) if isinstance(poem, str) else list(poem)
    rep = VenbaReport(valid=True)
    if len(words) != 15:
        rep.valid = False
        rep.errors.append(f"Expected 15 feet (4+4+4+3), got {len(words)}.")
        return rep

    prev = None
    for i, w in enumerate(words):
        asai = encode_asai(w)
        ok, msg = True, ""
        if i == EETRUCHEER:
            vp = eetrucheer_vaipadu(w)
            cat = "Eetrucheer" if vp else None
            if not vp:
                ok, msg = False, "final foot must be Naal/Malar/Kaasu/Pirappu"
        else:
            vp, cat = classify(asai)
            if cat not in VENBA_CATEGORIES:
                ok = False
                msg = f"{vp or asai} ({cat or 'unclassified'}) not allowed in Venba"
        th = None
        if prev is not None:
            th = thalai_name(prev, asai)
            if not is_vendalai(th):
                ok = False
                msg = (msg + "; " if msg else "") + f"link is {th}"
        rep.feet.append(FootReport(i, w, asai, split_display(w), vp, cat, th, ok, msg))
        rep.valid &= ok
        prev = asai

    a = [ethukai_key(words[i], strict_ethukai) for i in (0, 4, THANICHOL)]
    b = [ethukai_key(words[i], strict_ethukai) for i in (8, 12)]
    if None in a or len(set(a)) != 1:
        rep.valid = False
        rep.errors.append(f"Ethukai A mismatch across feet 0, 4, 7: {a}")
    if None in b or len(set(b)) != 1:
        rep.valid = False
        rep.errors.append(f"Ethukai B mismatch across feet 8, 12: {b}")
    if rep.valid:
        rep.vikarpam = "Oru Vikarpam" if a[0] == b[0] else "Iru Vikarpam"
    return rep
