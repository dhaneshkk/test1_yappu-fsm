"""The clingo (ASP) backend must agree with the Datalog engine, fact for fact and proof for proof."""
import importlib.util
import pathlib
import random
import sys

import pytest

clingo = pytest.importorskip("clingo")

from yappu_fsm import data_path
from yappu_fsm.rules import RuleCoder, compile_rules, parse_rules
from yappu_fsm.rules.asp import AspProgram, AspRuleCoder

RULES = pathlib.Path(data_path("icd10cm_rules.ace"))
_tools = pathlib.Path(__file__).resolve().parents[1] / "tools" / "export_tutorial_data.py"
_spec = importlib.util.spec_from_file_location("export_tutorial_data_asp", _tools)
_etd = importlib.util.module_from_spec(_spec)
sys.modules["export_tutorial_data_asp"] = _etd
_spec.loader.exec_module(_etd)
DEMO_NOTES = _etd.RULE_NOTES + _etd.NOTES


@pytest.fixture
def coders(linker):
    return RuleCoder(linker, RULES), AspRuleCoder(linker, RULES)


def _same(d, a):
    assert a.codes == d.codes
    assert list(a.blocked) == list(d.blocked) and a.blocked == d.blocked
    assert [(x, y) for x, y, _ in a.conflicts] == [(x, y) for x, y, _ in d.conflicts]
    assert a.proofs == d.proofs


@pytest.mark.parametrize("setting", ["outpatient", "inpatient"])
@pytest.mark.parametrize("note", DEMO_NOTES)
def test_asp_matches_datalog_on_demo_notes(coders, note, setting):
    _same(coders[0].code(note, setting), coders[1].code(note, setting))


def _random_note(rng, entities):
    cues = ["", "", "", "probable ", "no ", "history of ", "family history of ", "rule out ", "possible "]
    parts = [f"{rng.choice(cues)}{e}" for e in rng.sample(entities, rng.randint(1, 5))]
    return ". ".join(p[:1].upper() + p[1:] for p in parts) + "."


def test_asp_matches_datalog_fuzz(coders, linker):
    rng = random.Random(7)
    entities = sorted(linker.lexicon)
    for _ in range(300):
        note = _random_note(rng, entities)
        for setting in ("outpatient", "inpatient"):
            d, a = coders[0].code(note, setting), coders[1].code(note, setting)
            _same(d, a)
            assert set(a.facts) == set(d.facts)


def test_full_models_identical(coders, linker):
    d, a = coders
    for note in DEMO_NOTES:
        facts = d.facts_for(linker.process(note), "inpatient")
        assert d.program.run(facts).facts == a.program.run(facts).facts


def test_asp_matches_datalog_under_rule_ablation(linker):
    """Switching off any single rule changes both engines the same way."""
    rules = parse_rules(RULES.read_text(encoding="utf-8"), concepts=linker.lexicon.keys())
    for r in rules:
        sub = [x for x in rules if x.rid != r.rid]
        d, a = RuleCoder(linker, sub), AspRuleCoder(linker, sub)
        for note in _etd.RULE_NOTES:
            assert a.code(note, "inpatient").codes == d.code(note, "inpatient").codes, r.rid


def test_exactly_one_answer_set(linker):
    prog = compile_rules(parse_rules(RULES.read_text(encoding="utf-8"), concepts=linker.lexicon.keys()))
    asp = AspProgram(prog)
    facts = RuleCoder(linker, RULES).facts_for(linker.process(DEMO_NOTES[0]), "outpatient")
    assert len(asp._solve(facts)) == 1


def test_asp_text_is_plain_program(coders):
    text = coders[1].asp.to_text()
    assert "yappu_why_" not in text and 'code(P, "E11.22")' in text
    assert "yappu_why_" in coders[1].asp.to_text(provenance=True)


def test_alternatives_resolve_excludes1(coders):
    note = "T1DM noted; also T2DM per old records."
    d = coders[0].code(note)
    assert [(x, y) for x, y, _ in d.conflicts] == [("E10.9", "E11.9")]     # Datalog only flags it
    assert coders[1].alternatives(note) == [["E10.9"], ["E11.9"]]           # ASP lists both valid sets


def test_alternatives_without_conflict_is_the_single_result(coders):
    note = DEMO_NOTES[0]
    assert coders[1].alternatives(note) == [coders[1].code(note).codes]
