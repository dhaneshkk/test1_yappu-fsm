import time

import pytest

from yappu_fsm import data_path
from yappu_fsm.clinical import ClinicalLinker
from yappu_fsm.rules import AceError, DatalogError, RuleCoder, compile_rules, parse_rule, parse_rules
from yappu_fsm.rules.datalog import Literal, Program, Rule, Var


@pytest.fixture(scope="module")
def coder(lexicon):
    return RuleCoder(ClinicalLinker(lexicon), data_path("icd10cm_rules.ace"))


# ---- controlled-English parser ------------------------------------------------

def test_parse_and_paraphrase():
    r = parse_rule("If the patient has a hypertension and the patient has a chronic-kidney-disease "
                   "then the code I12.9 is assigned and the code I10 is not assigned.", "HTN")
    assert r.paraphrase() == ("IF { the patient has hypertension } AND { the patient has chronic kidney disease } "
                              "THEN { code I12.9 is assigned } AND { code I10 is blocked }")
    texts = [str(d) for d in r.to_datalog()]
    assert texts[0] == 'code(P, "I12.9") :- patient(P), has(P, hypertension), has(P, "chronic-kidney-disease").'
    assert texts[1].startswith('blocked(P, "I10")')


@pytest.mark.parametrize("sentence,expected", [
    ("If the patient has hypertension then the code I10 is assigned.", "'a', 'an', 'the' or a variable"),
    ("If the patient has a hypertension then the code I1X0 is assigned.", "an ICD-10-CM code"),
    ("If the patient had a hypertension then the code I10 is assigned.", "'has'"),
    ("If the patient has a hypertension then the code I10 is assigned", "'.'"),
    ("If the patient has a diagnosis X then the candidate code of Y is assigned.", "a variable introduced earlier"),
])
def test_parse_errors_are_precise(sentence, expected):
    with pytest.raises(AceError) as e:
        parse_rule(sentence)
    assert expected in str(e.value)


def test_unknown_concept_rejected(lexicon):
    with pytest.raises(AceError) as e:
        parse_rule("If the patient has a sepsis then the code A41.9 is assigned.", concepts=lexicon.keys())
    assert "concept from the lexicon" in str(e.value)


def test_all_condition_forms_parse(lexicon):
    rules = parse_rules("""
    A: If a patient has a history of a stroke then the code Z86.73 is assigned.
    B: If a relative of the patient has a colon-cancer then the code Z80.0 is assigned.
    C: If the patient does not have a fever and the patient has a cough then the code R05.9 is assigned.
    D: If the encounter is an outpatient-encounter and the patient has an uncertain pneumonia then the code R05.9 is assigned.
    E: If the patient has a knee-osteoarthritis and the knee-osteoarthritis is on both sides then the code M17.0 precedes the code I10.
    F: If the code E10.9 is assigned and the code E11.9 is not assigned then the codes E10.9 and E11.9 conflict.
    """, concepts=lexicon.keys())
    assert [r.rid for r in rules] == list("ABCDEF")
    compile_rules(rules)


def test_unstratifiable_rules_rejected():
    rules = parse_rules("If the code I10 is not assigned then the code I10 is assigned.")
    with pytest.raises(DatalogError, match="not stratifiable"):
        compile_rules(rules)


# ---- Datalog engine -------------------------------------------------------------

def test_datalog_recursion_and_negation():
    X, Y, Z = Var("X"), Var("Y"), Var("Z")
    prog = Program([
        Rule(Literal("path", (X, Y)), [Literal("edge", (X, Y))], "r1"),
        Rule(Literal("path", (X, Z)), [Literal("path", (X, Y)), Literal("edge", (Y, Z))], "r2"),
        Rule(Literal("cut", (X, Y)), [Literal("node", (X,)), Literal("node", (Y,)), Literal("path", (X, Y), True)], "r3"),
    ])
    m = prog.run([("edge", "a", "b"), ("edge", "b", "c"), ("edge", "c", "a")] + [("node", n) for n in "abcd"])
    assert len(m.query("path")) == 9           # full cycle on a, b, c
    assert ("cut", "a", "d") in m.facts and ("cut", "a", "b") not in m.facts
    assert m.explain(("path", "a", "a"))[0].startswith("path(a, a)  [r2]")


def test_unsafe_rule_rejected():
    X, Y = Var("X"), Var("Y")
    with pytest.raises(DatalogError, match="unsafe"):
        Program([Rule(Literal("p", (X, Y)), [Literal("q", (X,))], "bad")])


# ---- ICD-10-CM rule set -----------------------------------------------------------

def test_diabetes_ckd_combination(coder):
    r = coder.code("Pt with T2DM and CKD.")
    assert r.codes == ["E11.22", "N18.9"]
    assert "E11.9" in r.blocked and any("DM_CKD" in line for line in r.proofs["E11.22"])


def test_hypertensive_ckd_and_sequencing(coder):
    r = coder.code("Pt with T2DM and CKD. Also HTN.")
    assert r.codes == ["E11.22", "I12.9", "N18.9"]
    assert set(r.blocked) == {"E11.9", "I10"}


def test_hypertension_alone_unchanged(coder):
    assert coder.code("Uncontrolled HTN.").codes == ["I10"]


def test_excludes1_conflict_reported(coder):
    r = coder.code("T1DM noted; also T2DM per old records.")
    assert [(a, b) for a, b, _ in r.conflicts] == [("E10.9", "E11.9")]


def test_integral_symptoms_suppressed_only_when_diagnosis_coded(coder):
    confirmed = coder.code("Pneumonia confirmed on imaging. Cough and fever for three days.")
    assert confirmed.codes == ["J18.9"] and {"R05.9", "R50.9"} <= set(confirmed.blocked)
    uncertain = coder.code("Probable CAP. Cough and fever for three days.")          # Guideline IV.H
    assert uncertain.codes == ["R05.9", "R50.9"]
    inpatient = coder.code("Probable CAP. Cough and fever for three days.", setting="inpatient")  # II.H
    assert inpatient.codes == ["J18.9"]


def test_known_limitation_trigger_scope_covers_symptoms(coder):
    # 'probable' scopes over 'cough and fever' in the same clause, so they are UNCERTAIN too.
    r = coder.code("Probable CAP with cough and fever.")
    assert r.codes == []


def test_inpatient_rule_matches_linker_inpatient_setting(coder, lexicon):
    """The ACE rule INPT_UNCERTAIN reproduces the hard-coded Guideline II.H behaviour."""
    import json
    lk = ClinicalLinker(lexicon, setting="inpatient")
    for line in open(data_path("gold_assertions.jsonl"), encoding="utf-8"):
        note = json.loads(line)["note"]
        expected = set(lk.allowed_codes(lk.process(note)))
        got = set(coder.code(note, "inpatient").codes) | set(coder.code(note, "inpatient").blocked)
        assert expected <= got, note


def test_rule_engine_speed(coder):
    notes = ["Pt with T2DM and CKD. Also HTN.", "Probable CAP. Cough and fever for three days."] * 50
    t = time.perf_counter()
    for n in notes:
        coder.code(n, "inpatient")
    assert (time.perf_counter() - t) / len(notes) < 0.05       # generous bound for slow CI machines


def test_datalog_export(coder):
    text = coder.program.to_text()
    assert "% DM_CKD.1:" in text and 'final(P, C) :- code(P, C), not blocked(P, C).' in text


# ---- sequencing -------------------------------------------------------------------

def test_derived_code_takes_place_of_replaced_code(coder):
    # E11.22 replaces E11.9 (1st finding), I12.9 replaces I10 (3rd); N18.9 follows both;
    # the symptom code goes after the diagnoses
    r = coder.code("Pt with T2DM and CKD. Also HTN. Probable CAP. Cough for three days.")
    assert r.codes == ["E11.22", "I12.9", "N18.9", "R05.9"]


def test_symptoms_and_factors_sequenced_last(coder):
    r = coder.code("Cough for three days. Family history of HTN. Probable CAP. Pt with T2DM and CKD.")
    assert r.codes == ["E11.22", "N18.9", "R05.9", "Z82.49"]
    r = coder.code("History of CVA. Headache. Left knee osteoarthritis.")
    assert r.codes == ["M17.12", "R51.9", "Z86.73"]


def test_inpatient_derived_code_sits_at_its_finding(coder):
    r = coder.code("Cough for three days. Probable CAP. Fever. HTN.", setting="inpatient")
    assert r.codes == ["J18.9", "I10"] and {"R05.9", "R50.9"} <= set(r.blocked)


def test_code_variables_and_kinds_parse():
    r = parse_rule("If a code C is assigned and the code C is a symptom code and a code D is assigned "
                   "and the code D is a diagnosis code then the code D precedes the code C.")
    assert r.paraphrase().endswith("THEN { code D is sequenced before code C }")
    with pytest.raises(AceError, match="only sequencing may use a code variable"):
        parse_rule("If a code C is assigned then the code C is not assigned.")
