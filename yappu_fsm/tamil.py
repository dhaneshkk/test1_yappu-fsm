"""Tamil orthography and prosody primitives.

Everything here is pure Python (no third-party dependencies):

* ``graphemes``       - split a word into visual letters (consonant + vowel sign / pulli).
* ``letter_classes``  - classify letters as Kuril (K), Nedil (N) or Otru (O).
* ``encode_asai``     - reduce a word to its binary Asai string ('1' = Ner, '0' = Nirai).
* ``ethukai_key``     - 2nd-letter rhyme key (consonant of the 2nd letter + length of the 1st).
* ``monai_key``       - 1st-letter alliteration class using the traditional Ina Ezhuthu groups.
* ``encode_english``  - rough English syllable-weight heuristic (approximate; see docstring).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

PULLI = "்"  # ்
AU_LENGTH_MARK = "ௗ"
AYTHAM = "ஃ"  # ஃ

# Dependent vowel signs
NEDIL_SIGNS = {"ா", "ீ", "ூ", "ே", "ை", "ோ", "ௌ"}  # ா ீ ூ ே ை ோ ௌ
KURIL_SIGNS = {"ி", "ு", "ெ", "ொ"}  # ி ு ெ ொ
COMBINING = NEDIL_SIGNS | KURIL_SIGNS | {PULLI, AU_LENGTH_MARK}

INDEPENDENT_KURIL = {"அ", "இ", "உ", "எ", "ஒ"}
INDEPENDENT_NEDIL = {"ஆ", "ஈ", "ஊ", "ஏ", "ஐ", "ஓ", "ஔ"}

# Hard consonants (வல்லினம்) – used for kutriyalukaram (Kaasu / Pirappu endings)
VALLINAM = {"க", "ச", "ட", "த", "ப", "ற"}

TAMIL_RE = re.compile(r"[஀-௿]")


def is_consonant(ch: str) -> bool:
    return "க" <= ch <= "ஹ"


def normalize(text: str) -> str:
    """NFC-normalise so decomposed ொ/ோ/ௌ become single code points."""
    return unicodedata.normalize("NFC", text)


def is_tamil(text: str) -> bool:
    return bool(TAMIL_RE.search(text))


def graphemes(word: str) -> List[str]:
    """Split a Tamil word into visual letters.

    >>> graphemes("வண்ண")
    ['வ', 'ண்', 'ண']
    """
    letters: List[str] = []
    for ch in normalize(word):
        if ch in COMBINING and letters:
            letters[-1] += ch
        elif TAMIL_RE.match(ch):
            letters.append(ch)
    return letters


def letter_class(letter: str) -> Optional[str]:
    """Return 'K' (kuril), 'N' (nedil), 'O' (otru / aytham) or None."""
    if not letter:
        return None
    base = letter[0]
    if base in INDEPENDENT_KURIL:
        return "K"
    if base in INDEPENDENT_NEDIL:
        return "N"
    if base == AYTHAM:
        return "O"
    if is_consonant(base):
        sign = letter[1:2]
        if sign == PULLI:
            return "O"
        if sign in NEDIL_SIGNS:
            return "N"
        return "K"  # bare consonant (inherent அ) or short vowel sign
    return None


def letter_classes(word: str) -> List[str]:
    return [c for c in (letter_class(g) for g in graphemes(word)) if c]


def asai_units(word: str) -> List[Tuple[str, str]]:
    """Split a word into Asai units, returning (text, '1'|'0') pairs.

    Rules (யாப்பிலக்கணம்):
      * a Nedil always closes its own Ner asai;
      * a Kuril followed by a Kuril or a Nedil forms a Nirai asai;
      * a lone Kuril forms a Ner asai;
      * trailing Otru letters attach to the preceding asai.
    """
    gs = graphemes(word)
    cls = [letter_class(g) for g in gs]
    units: List[Tuple[str, str]] = []
    i, n = 0, len(gs)
    while i < n:
        c = cls[i]
        if c == "O":  # stray otru at word start – attach to previous or skip
            if units:
                units[-1] = (units[-1][0] + gs[i], units[-1][1])
            i += 1
            continue
        if c == "K" and i + 1 < n and cls[i + 1] in ("K", "N"):
            text, kind, i = gs[i] + gs[i + 1], "0", i + 2
        else:
            text, kind, i = gs[i], "1", i + 1
        while i < n and cls[i] == "O":
            text += gs[i]
            i += 1
        units.append((text, kind))
    return units


def encode_asai(word: str) -> str:
    """Binary Asai string for a Tamil word.

    >>> encode_asai("கூடினால்")   # கூ / டினால்  -> Koovilam
    '10'
    """
    return "".join(kind for _, kind in asai_units(word))


def split_display(word: str) -> str:
    """Human-readable asai split, e.g. 'கூ/டினால்'."""
    return "/".join(t for t, _ in asai_units(word))


def has_kutriyalukaram_ending(word: str) -> bool:
    """True when the word ends in a short final உ on a hard consonant (கு சு டு து பு று)."""
    gs = graphemes(word)
    if len(gs) < 2:
        return False
    last = gs[-1]
    return len(last) == 2 and last[0] in VALLINAM and last[1] == "ு"


# --------------------------------------------------------------------------
# Ethukai (எதுகை) – 2nd-letter rhyme
# --------------------------------------------------------------------------


def ethukai_key(word: str, strict: bool = False) -> Optional[str]:
    """Key used to compare Ethukai between line-initial words.

    Default (traditional) rule: the 2nd letters share the same consonant and the
    1st letters have the same length (both kuril or both nedil).  ``strict=True``
    compares the full 2nd letter instead of its consonant.
    """
    gs = graphemes(word)
    if len(gs) < 2:
        return None
    first_len = "N" if letter_class(gs[0]) == "N" else "K"
    second = gs[1] if strict else gs[1][0]
    return f"{first_len}:{second}"


def ethukai_match(a: str, b: str, strict: bool = False) -> bool:
    ka, kb = ethukai_key(a, strict), ethukai_key(b, strict)
    return ka is not None and ka == kb


# --------------------------------------------------------------------------
# Monai (மோனை) – 1st-letter alliteration with Ina Ezhuthu equivalence
# --------------------------------------------------------------------------

_VOWEL_GROUP = {
    "அ": "A", "ஆ": "A", "ஐ": "A", "ஔ": "A",
    "இ": "I", "ஈ": "I", "எ": "I", "ஏ": "I",
    "உ": "U", "ஊ": "U", "ஒ": "U", "ஓ": "U",
}
_SIGN_GROUP = {
    "": "A", "ா": "A", "ை": "A", "ௌ": "A",
    "ி": "I", "ீ": "I", "ெ": "I", "ே": "I",
    "ு": "U", "ூ": "U", "ொ": "U", "ோ": "U",
}
# Traditional consonant Monai pairs: ச/த, ஞ/ந, ம/வ
_CONSONANT_GROUP = {"ச": "சத", "த": "சத", "ஞ": "ஞந", "ந": "ஞந", "ம": "மவ", "வ": "மவ"}


def monai_key(word: str) -> Optional[str]:
    if not is_tamil(word):
        m = re.search(r"[a-zA-Z]", word)
        return m.group(0).lower() if m else None
    gs = graphemes(word)
    if not gs:
        return None
    first = gs[0]
    if first[0] in _VOWEL_GROUP:
        return "V:" + _VOWEL_GROUP[first[0]]
    if is_consonant(first[0]):
        cons = _CONSONANT_GROUP.get(first[0], first[0])
        return f"C:{cons}:{_SIGN_GROUP.get(first[1:2], 'A')}"
    return None


# --------------------------------------------------------------------------
# English (approximate)
# --------------------------------------------------------------------------


def encode_english(word: str) -> str:
    """Very rough English syllable-weight heuristic mapped onto Ner/Nirai.

    Heavy syllables (long vowel digraph or ≥4 letters) -> '1', light -> '0';
    runs of light syllables collapse to one '0'.  This is an illustration of the
    idea only; real English meter needs a pronouncing dictionary (e.g. CMUdict).
    """
    clean = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not clean:
        return ""
    syllables = re.findall(r"[^aeiouy]*[aeiouy]+(?:[^aeiouy](?![aeiouy]))*", clean) or [clean]
    weights = ["1" if re.search(r"[aeiou]{2}|[aeiou][a-z]e$", s) or len(s) >= 4 else "0" for s in syllables]
    return re.sub(r"00+", "0", "".join(weights))[:3]


def encode(word: str) -> str:
    """Asai for Tamil words; approximate weight pattern for English words."""
    return encode_asai(word) if is_tamil(word) else encode_english(word)
