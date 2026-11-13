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


# ---- ICD-10-CM coding settings and expanded lexicon ---------------------------

def test_outpatient_vs_inpatient_uncertain(lexicon):
    from yappu_fsm.clinical import ClinicalLinker
    note = "Probable pneumonia, start antibiotics."
    (out,) = ClinicalLinker(lexicon).process(note)
    (inp,) = ClinicalLinker(lexicon, setting="inpatient").process(note)
    assert out.status == "DO_NOT_CODE"                      # CM Section IV.H
    assert inp.codes == ["J18.9"] and inp.assertion == "UNCERTAIN"   # CM Section II.H


def test_inpatient_still_blocks_negation(lexicon):
    from yappu_fsm.clinical import ClinicalLinker
    (f,) = ClinicalLinker(lexicon, setting="inpatient").process("Denies fever.")
    assert f.status == "DO_NOT_CODE"


def test_invalid_setting(lexicon):
    from yappu_fsm.clinical import ClinicalLinker
    with pytest.raises(ValueError):
        ClinicalLinker(lexicon, setting="er")


@pytest.mark.parametrize("note,entity,codes", [
    ("Hx of AFib, on apixaban.", "atrial fibrillation", ["I48.91"]),
    ("CHF exacerbation.", "heart failure", ["I50.9"]),
    ("History of CVA in 2020.", "stroke", ["Z86.73"]),
    ("Father had colon cancer.", "colon cancer", ["Z80.0"]),
    ("Rt CTS confirmed on EMG.", "carpal tunnel syndrome", ["G56.01"]),
    ("Bilateral carpal tunnel syndrome.", "carpal tunnel syndrome", ["G56.03"]),
    ("Left shoulder pain after fall.", "shoulder pain", ["M25.512"]),
    ("Former smoker, history of nicotine dependence.", "nicotine dependence", ["Z87.891"]),
    ("UTI, treated with nitrofurantoin.", "urinary tract infection", ["N39.0"]),
])
def test_expanded_cm_lexicon(linker, note, entity, codes):
    f = by_entity(linker.process(note))[entity]
    assert f.codes == codes


# ---- index ------------------------------------------------------------------------

def test_index_matches_full_scan(lexicon):
    import json
    import random

    from yappu_fsm import data_path
    from yappu_fsm.clinical import ClinicalLinker
    fast, slow = ClinicalLinker(lexicon, index=True), ClinicalLinker(lexicon, index=False)
    notes = [json.loads(l)["note"] for l in open(data_path("gold_assertions.jsonl"), encoding="utf-8")]
    rng = random.Random(7)
    for _ in range(300):  # random single-edit typos of lexicon terms inside a sentence
        term = list(rng.choice(list(lexicon)))
        i = rng.randrange(len(term))
        op = rng.choice("sdit")
        if op == "s":
            term[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        elif op == "d" and len(term) > 1:
            del term[i]
        elif op == "i":
            term.insert(i, rng.choice("aeiou"))
        elif op == "t" and i + 1 < len(term):
            term[i], term[i + 1] = term[i + 1], term[i]
        notes.append("Patient has " + "".join(term) + " today.")
    for n in notes:
        assert [f.to_dict() for f in fast.process(n)] == [f.to_dict() for f in slow.process(n)], n


def test_index_scales(lexicon):
    import random
    import string
    import time

    from yappu_fsm.clinical import ClinicalLinker
    rng = random.Random(0)
    big = dict(lexicon)
    while len(big) < 50_000:
        big[" ".join("".join(rng.choices(string.ascii_lowercase, k=rng.randint(5, 11)))
                     for _ in range(rng.randint(1, 4)))] = {"PRESENT_UNSPECIFIED": "Z99.9"}
    lk = ClinicalLinker(big)
    note = "Pt has lt knee osteoarthritus and mild NPDR OD. Denies SOB, but has uncontrolled HTN. R/O CAP."
    lk.process(note)
    t = time.perf_counter()
    for _ in range(20):
        found = lk.process(note)
    assert (time.perf_counter() - t) / 20 < 0.05
    assert [f.codes for f in found if f.status == "CODE"] == [["M17.12"], ["E11.3291"], ["I10"]]
