"""NegEx / ConText behaviour of the clinical linker."""
import pytest

from yappu_fsm import data_path
from yappu_fsm.clinical import evaluate


def assertion_of(linker, note, entity):
    return {f.entity: f.assertion for f in linker.process(note)}[entity]


@pytest.mark.parametrize("note,entity,state", [
    ("Ruled out pneumonia on admission.", "pneumonia", "NEGATED"),                       # [PREN] ruled out
    ("Pneumonia in the left lower lobe was ruled out.", "pneumonia", "NEGATED"),         # [POST] 5 tokens back
    ("No history of MI.", "myocardial infarction", "NEGATED"),                           # longest trigger wins
    ("No fever aside from mild cough.", "cough", "PRESENT"),                             # [CONJ] ends scope
    ("No fever aside from mild cough.", "fever", "NEGATED"),
    ("Absence of chest pain at rest.", "chest pain", "NEGATED"),
    ("Headache was not demonstrated on review.", "headache", "NEGATED"),
    ("Fever never developed after surgery.", "fever", "NEGATED"),
    ("Return if fever develops.", "fever", "HYPOTHETICAL"),                              # ConText hypothetical
    ("If the patient develops chest pain, call 911.", "chest pain", "HYPOTHETICAL"),
    ("No change in hypertension since last visit.", "hypertension", "PRESENT"),          # [PSEU]
    ("Pneumonia cannot be ruled out.", "pneumonia", "UNCERTAIN"),
    ("Not rule out pneumonia yet, start empiric therapy.", "pneumonia", "UNCERTAIN"),
    ("The patient was evaluated for stroke, which was ruled out.", "stroke", "NEGATED"),
    ("Cough present; fever absent.", "fever", "NEGATED"),
    ("Shortness of breath due to heart failure, without chest pain.", "heart failure", "PRESENT"),
])
def test_negex_cases(linker, note, entity, state):
    assert assertion_of(linker, note, entity) == state


def test_hypothetical_never_coded(linker):
    (f,) = linker.process("Return if fever develops.")
    assert f.status == "DO_NOT_CODE"


def test_scope_window_limit(linker):
    # a pre-trigger reaches at most 6 tokens
    note = "No acute distress noted by the team during the visit today with hypertension."
    assert assertion_of(linker, note, "hypertension") == "PRESENT"


@pytest.mark.parametrize("gold,floor", [("gold_assertions.jsonl", 0.95), ("gold_negex.jsonl", 0.99),
                                        ("gold_negex_heldout.jsonl", 0.99)])
def test_gold_sets(linker, gold, floor):
    assert evaluate(linker, data_path(gold))["assertion_mcc"] >= floor
