from conftest import NANNERI, POEM_1, POEM_2

from yappu_fsm.prosody import eetrucheer_vaipadu, format_venba, parse_venba, thalai_name, validate_nerisai_venba


def test_three_poems_valid():
    for p in (POEM_1, POEM_2, NANNERI):
        rep = validate_nerisai_venba(p)
        assert rep.valid, rep.lines()
        assert rep.vikarpam == "Iru Vikarpam"


def test_corrected_thalai_labels_poem1():
    rep = validate_nerisai_venba(POEM_1)
    labels = {f.word: f.thalai_in for f in rep.feet}
    assert labels["பண்ணும்"] == "Iyarseer Vendalai"     # after Koovilam கூடினால்
    assert labels["கண்ணும்"] == "Iyarseer Vendalai"     # after Koovilam பார்ப்பவர்
    assert labels["நம்மொழி"] == "Iyarseer Vendalai"     # after Karuvilam இலக்கணம்
    assert labels["கூடினால்"] == "Venseer Vendalai"     # after Kaycheer இசையோடு


def test_kaasu_eetrucheer():
    assert eetrucheer_vaipadu("கொக்கு") == "Kaasu"
    assert eetrucheer_vaipadu("பிறப்பு") == "Pirappu"
    assert eetrucheer_vaipadu("மதி") == "Malar"
    assert eetrucheer_vaipadu("நாள்") == "Naal"
    assert eetrucheer_vaipadu("வானில்") is None


def test_thalai_break_detected():
    words = parse_venba(POEM_2)
    words[1] = "வானில்"  # Thema after Thema -> Nerondru Aasiriyathalai
    rep = validate_nerisai_venba(words)
    assert not rep.valid
    assert any("Aasiriyathalai" in f.message for f in rep.feet)


def test_kanicheer_rejected():
    assert thalai_name("110", "1") == "Ondraadha Vanjithalai"
    words = parse_venba(POEM_2)
    words[2] = "கூவிளங்கனி"  # 1-0-0 Kanicheer
    assert not validate_nerisai_venba(words).valid


def test_ethukai_mismatch_and_count():
    words = parse_venba(POEM_2)
    words[12] = "நிலவொளி"
    rep = validate_nerisai_venba(words)
    assert not rep.valid and any("Ethukai B" in e for e in rep.errors)
    assert not validate_nerisai_venba(words[:14]).valid


def test_strict_ethukai_rejects_nanneri():
    assert not validate_nerisai_venba(NANNERI, strict_ethukai=True).valid


def test_format_roundtrip():
    words = parse_venba(POEM_2)
    assert parse_venba(format_venba(words)) == words
