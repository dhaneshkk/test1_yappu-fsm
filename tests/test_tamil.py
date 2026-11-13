import pytest

from yappu_fsm.tamil import (encode_asai, encode_english, ethukai_key, ethukai_match, graphemes,
                             has_kutriyalukaram_ending, letter_classes, monai_key, split_display)


def test_graphemes():
    assert graphemes("வண்ண") == ["வ", "ண்", "ண"]
    assert graphemes("இருளினை") == ["இ", "ரு", "ளி", "னை"]


def test_decomposed_vowel_signs_are_normalised():
    decomposed = "கொக்கு"  # கொக்கு with ொ written as ெ + ா
    assert encode_asai(decomposed) == encode_asai("கொக்கு") == "11"


def test_letter_classes():
    assert letter_classes("கால்") == ["N", "O"]
    assert letter_classes("கடாம்") == ["K", "N", "O"]


@pytest.mark.parametrize("word,asai", [
    ("க", "1"), ("கல்", "1"), ("கா", "1"), ("கால்", "1"),
    ("கட", "0"), ("கடல்", "0"), ("கடா", "0"), ("கடாம்", "0"),
    ("எண்ணும்", "11"), ("எழுத்தும்", "01"), ("நிலவொளி", "00"), ("இசையோடு", "011"),
    # words the original PDF / thread split wrongly
    ("கூடினால்", "10"), ("பார்ப்பவர்", "10"), ("இலக்கணம்", "00"),
    ("தவழ்ந்திட", "00"), ("ஏற்றிடும்", "10"), ("யிருக்குமாம்", "00"),
    ("ரறிவிலரென்", "001"), ("விண்மீன்கள்", "111"), ("மதி", "0"),
])
def test_asai(word, asai):
    assert encode_asai(word) == asai


def test_split_display():
    assert split_display("கூடினால்") == "கூ/டினால்"
    assert split_display("இலக்கணம்") == "இலக்/கணம்"


def test_kutriyalukaram():
    assert has_kutriyalukaram_ending("கொக்கு")
    assert has_kutriyalukaram_ending("பிறப்பு")
    assert not has_kutriyalukaram_ending("மதி")


def test_ethukai_consonant_rule():
    # Nanneri: அடக்க / கடக்கக் / மடைத்தலையில் share ட ; ஓடுமீ / வாடி share ட with nedil first letters
    assert ethukai_match("அடக்க", "மடைத்தலையில்")
    assert ethukai_match("ஓடுமீ", "வாடி")
    assert not ethukai_match("அடக்க", "மடைத்தலையில்", strict=True)
    # first-letter length must agree
    assert not ethukai_match("அடக்க", "வாடி")
    assert ethukai_match("அமுதத்", "நம்மொழி")


def test_monai_ina_ezhuthu():
    assert monai_key("வண்ண") == monai_key("மண்ணில்")      # ம/வ
    assert monai_key("தமிழ்") == monai_key("சங்கம்")       # ச/த
    assert monai_key("இனிய") == monai_key("எண்ணும்")      # இ/எ
    assert monai_key("கடல்") != monai_key("கிளி")         # different vowel group


def test_english_heuristic_shape():
    for w in ["moonlight", "melody", "a", "strengths"]:
        code = encode_english(w)
        assert set(code) <= {"0", "1"} and 1 <= len(code) <= 3
