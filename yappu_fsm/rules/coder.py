"""Note -> linker facts -> Datalog (compiled from controlled-English rules) -> codes with proofs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..clinical.linker import ClinicalLinker, Finding
from .ace import AceRule, compile_rules, parse_rules
from .datalog import Fact, Program

PATIENT = "patient"
SIDE = {"RIGHT": "right", "LEFT": "left", "BILATERAL": "both"}
PRED = {"PRESENT": "has", "UNCERTAIN": "uncertain", "HISTORICAL": "history",
        "FAMILY_HISTORY": "family", "NEGATED": "negated"}


def concept(entity: str) -> str:
    return entity.replace(" ", "-")


@dataclass
class CodingResult:
    codes: List[str]                                  # final, sequenced
    blocked: Dict[str, List[str]]                     # code -> proof lines of the block
    conflicts: List[Tuple[str, str, List[str]]]       # (code, code, proof)
    proofs: Dict[str, List[str]]                      # final code -> proof lines
    findings: List[Finding]
    facts: List[Fact]
    setting: str

    def report(self) -> str:
        out = [f"setting: {self.setting}", "final codes: " + (", ".join(self.codes) or "(none)")]
        for c in self.codes:
            out.append(f"\n[{c}]")
            out += ["  " + l for l in self.proofs[c]]
        for c, proof in self.blocked.items():
            out.append(f"\nblocked {c}:")
            out += ["  " + l for l in proof]
        for a, b, proof in self.conflicts:
            out.append(f"\nCONFLICT {a} / {b} (coder review):")
            out += ["  " + l for l in proof]
        return "\n".join(out)


class RuleCoder:
    """Combine the fast rule-based linker (facts) with controlled-English coding rules (Datalog).

    The linker always runs in outpatient mode (uncertain diagnoses produce no code);
    the encounter setting becomes the fact ``setting(patient, outpatient|inpatient)``
    and rules such as INPT_UNCERTAIN decide what to do with uncertain findings.
    """

    def __init__(self, linker: ClinicalLinker, rules: Union[str, Path, Sequence[AceRule]]):
        self.linker = linker if linker.setting == "outpatient" else ClinicalLinker(linker.lexicon)
        if isinstance(rules, (str, Path)) and Path(str(rules)).exists():
            rules = Path(rules).read_text(encoding="utf-8")
        if isinstance(rules, str):
            rules = parse_rules(rules, concepts=self.linker.lexicon.keys())
        self.rules: List[AceRule] = list(rules)
        self.program: Program = compile_rules(self.rules)
        self.rules_by_id = {r.rid: r for r in self.program.rules}

    def _candidates(self, f: Finding) -> List[str]:
        cmap = self.linker.lexicon.get(f.entity, {})
        val = cmap.get(f"PRESENT_{f.laterality}") or cmap.get("PRESENT_UNSPECIFIED") or cmap.get("PRESENT")
        return [] if val is None else ([val] if isinstance(val, str) else list(val))

    def facts_for(self, findings: Sequence[Finding], setting: str) -> List[Fact]:
        facts: List[Fact] = [("patient", PATIENT), ("setting", PATIENT, setting)]
        for f in findings:
            c = concept(f.entity)
            facts.append((PRED[f.assertion], PATIENT, c))
            if f.laterality in SIDE:
                facts.append(("side", PATIENT, c, SIDE[f.laterality]))
            for code in self._candidates(f):
                facts.append(("candidate", PATIENT, c, code))
            for code in f.codes:                     # codes the linker assigned directly
                facts.append(("code", PATIENT, code))
        seen, out = set(), []
        for x in facts:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def code(self, note: str, setting: str = "outpatient") -> CodingResult:
        if setting not in ("outpatient", "inpatient"):
            raise ValueError("setting must be 'outpatient' or 'inpatient'")
        findings = self.linker.process(note)
        facts = self.facts_for(findings, setting)
        m = self.program.run(facts)
        explain = lambda f: m.explain(f, self.rules_by_id, hide=("patient",))

        order: List[str] = []
        for f in facts + [x for x in m.query("code")]:
            if f[0] == "code" and f[2] not in order:
                order.append(f[2])
        final = [c for c in order if ("final", PATIENT, c) in m.facts]
        final = _sequence(final, [(f[2], f[3]) for f in m.query("precedes")])
        blocked = {f[2]: explain(f) for f in m.query("blocked") if ("code", PATIENT, f[2]) in m.facts}
        conflicts = [(f[2], f[3], explain(f)) for f in m.query("conflict")]
        proofs = {c: explain(("final", PATIENT, c)) for c in final}
        return CodingResult(final, blocked, conflicts, proofs, findings, facts, setting)


def _sequence(codes: List[str], before: List[Tuple[str, str]]) -> List[str]:
    """Stable topological order honouring 'X precedes Y' where both are present."""
    present = set(codes)
    edges = [(a, b) for a, b in before if a in present and b in present and a != b]
    out, remaining = [], list(codes)
    while remaining:
        for c in remaining:
            if not any(b == c and a in remaining for a, b in edges):
                out.append(c)
                remaining.remove(c)
                break
        else:  # cycle: keep original order for the rest
            out += remaining
            break
    return out
