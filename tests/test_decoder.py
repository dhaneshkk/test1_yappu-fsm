"""Properties of the pure-Python Venba decoder (no torch needed)."""
import random

import pytest

from yappu_fsm.decoding import ICD10CodeTrie, VenbaDecoder
from yappu_fsm.lexicon import VenbaLexicon
from yappu_fsm.prosody import validate_nerisai_venba


def random_poem(dec, rng):
    s = dec.initial()
    while True:
        cands = dec.lex.candidates(s.k, s.expected, s.key_a, s.key_b)
        assert cands, f"dead end at foot {s.k}"
        s = dec.advance(s, rng.choice(cands).word.encode())
        if dec.is_complete(s):
            return dec.finished_words(s)
        s = dec.advance(s, b" ")
        assert s is not None


@pytest.mark.parametrize("scheme", ["iru", "oru"])
def test_every_decoder_path_is_a_valid_venba(scheme):
    dec = VenbaDecoder(VenbaLexicon.sample(scheme=scheme))
    rng = random.Random(42)
    for _ in range(200):
        words = random_poem(dec, rng)
        rep = validate_nerisai_venba(words)
        assert rep.valid, rep.lines()
        if scheme == "oru":
            assert rep.vikarpam == "Oru Vikarpam"


def test_decoder_rejects_invalid_continuations():
    dec = VenbaDecoder(VenbaLexicon.sample())
    s = dec.advance(dec.initial(), "எண்ணும்".encode())
    s = dec.advance(s, b" ")
    assert dec.advance(s, "வானில்".encode()) is None   # Thema after Thema breaks Vendalai
    assert dec.advance(s, "எழுத்தும்".encode()) is not None
    assert dec.advance(s, b" ") is None                   # double space


def test_partial_utf8_bytes_are_accepted():
    dec = VenbaDecoder(VenbaLexicon.sample())
    b = "எண்ணும்".encode()
    s = dec.advance(dec.initial(), b[:2])                 # half a code point
    assert s is not None
    assert dec.advance(s, b[2:]) is not None


def test_leading_space_allowed_once():
    dec = VenbaDecoder(VenbaLexicon.sample())
    s = dec.advance(dec.initial(), b" ")
    assert s is not None and dec.advance(s, b" ") is None


def test_icd_trie():
    t = ICD10CodeTrie(["E11.3291", "E11.3292", "I10"])
    assert t.is_code("I10") and not t.is_code("I1")
    assert t.walk("E11.329") is not None and t.walk("E11.33") is None
