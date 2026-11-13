import pytest

from yappu_fsm.clinical import clinical_phonetic_key


def by_entity(findings):
    return {f.entity: f for f in findings}


# ---- laterality ---------------------------------------------------------------

def test_multi_laterality_no_cross_contamination(linker):
    r = by_entity(linker.process("Pt presents with mild NPDR OD and lt knee osteoarthritis."))
    assert r["mild nonproliferative diabetic retinopathy"].laterality == "RIGHT"
    assert r["mild nonproliferative diabetic retinopathy"].codes == ["E11.3291"]
    assert r["knee osteoarthritis"].laterality == "LEFT"
    assert r["knee osteoarthritis"].codes == ["M17.12"]


def test_pre_and_post_laterality(linker):
    r = by_entity(linker.process("Right hip pain noted along with mild NPDR OS."))
    assert r["hip pain"].codes == ["M25.551"]
    assert r["mild nonproliferative diabetic retinopathy"].codes == ["E11.3292"]


def test_bilateral(linker):
    r = by_entity(linker.process("Exam shows b/l knee osteoarthritis and mild NPDR OU."))
    assert r["knee osteoarthritis"].codes == ["M17.0"]
    assert r["mild nonproliferative diabetic retinopathy"].codes == ["E11.3293"]


def test_bilateral_hip_pain_codes_both_sides(linker):
    (f,) = linker.process("Bilateral hip pain worse on stairs.")
    assert f.codes == ["M25.551", "M25.552"]


def test_laterality_distance_threshold(linker):
    (f,) = linker.process("Right arm scratch noted on skin exam yesterday and patient has hypertension.")
    assert f.entity == "hypertension" and f.laterality == "UNSPECIFIED" and f.codes == ["I10"]


# ---- assertion ----------------------------------------------------------------

def test_wall_resets_negation(linker):
    r = by_entity(linker.process("Patient denies SOB, but has uncontrolled HTN."))
    assert r["shortness of breath"].status == "DO_NOT_CODE"
    assert r["hypertension"].codes == ["I10"]


def test_nearest_trigger_wins(linker):
    # the thread's version marked SOB as HISTORICAL here
    r = by_entity(linker.process("Denies SOB, history of MI."))
    assert r["shortness of breath"].assertion == "NEGATED"
    assert r["myocardial infarction"].assertion == "HISTORICAL"
    assert r["myocardial infarction"].codes == ["I25.2"]


def test_trigger_scope_limit(linker):
    (f,) = linker.process("No known drug allergies, takes lisinopril for longstanding hypertension.")
    assert f.assertion == "PRESENT"


def test_post_triggers(linker):
    (f,) = linker.process("Chest X-ray clear; pneumonia ruled out.")
    assert f.assertion == "NEGATED"
    (g,) = linker.process("Pneumonia has resolved after treatment.")
    assert g.assertion == "HISTORICAL"


def test_uncertain_and_family_history(linker):
    (f,) = linker.process("R/O CAP.")
    assert f.assertion == "UNCERTAIN" and f.status == "DO_NOT_CODE"
    r = by_entity(linker.process("Family history of HTN. Mother had type 2 diabetes."))
    assert r["hypertension"].codes == ["Z82.49"]
    assert r["type 2 diabetes"].codes == ["Z83.3"]


def test_negated_with_laterality_still_blocked(linker):
    (f,) = linker.process("Patient denies right hip pain today.")
    assert f.laterality == "RIGHT" and f.status == "DO_NOT_CODE"


def test_pseudo_trigger(linker):
    (f,) = linker.process("No change in hypertension since last visit.")
    assert f.assertion == "PRESENT"


def test_case_sensitive_abbreviations(linker):
    assert linker.process("Bruise over the knee cap.") == []
    assert linker.process("Possible CAP.")[0].entity == "pneumonia"


def test_decimal_does_not_split_clause(linker):
    assert len(linker.split_clauses("HbA1c 7.5 today. Denies SOB")) == 2


# ---- fuzzy matching -----------------------------------------------------------

@pytest.mark.parametrize("a,b", [("osteoarthritis", "osteoarthritus"), ("pneumonia", "pnemonia"),
                                 ("nephropathy", "nefropathy"), ("hypertension", "hypertensn"),
                                 ("retinopathy", "retinopthy")])
def test_phonetic_keys(a, b):
    assert clinical_phonetic_key(a) == clinical_phonetic_key(b)


def test_typo_recovery_end_to_end(linker):
    r = by_entity(linker.process(
        "Pt has lt knee osteoarthritus and diabetic retinopthy. Denies pnemonia, but has uncontrolled hypertensn."))
    assert r["knee osteoarthritis"].match_type == "PHONETIC_FUZZY" and r["knee osteoarthritis"].codes == ["M17.12"]
    assert r["diabetic retinopathy"].codes == ["E11.319"]
    assert r["pneumonia"].status == "DO_NOT_CODE"
    assert r["hypertension"].codes == ["I10"]


def test_transposition_typo(linker):
    (f,) = linker.process("Patient presents with uncontrolled hypertesnion.")
    assert f.entity == "hypertension" and f.match_type == "EDIT_FUZZY"


def test_short_word_guardrail(linker):
    # the thread's test failed here: 'hip rain' vs 'hip pain' similarity is 0.875
    assert linker.process("Walked in the hip rain yesterday with no complaints.") == []


def test_allowed_codes_dedup(linker):
    f = linker.process("HTN. Also hypertension noted.")
    assert linker.allowed_codes(f) == ["I10"]
