"""Answer Set Programming backend (clingo) for the controlled-English coding rules.

The ACE rules compile to one :class:`~yappu_fsm.rules.datalog.Program`. That program is
stratified, so as an ASP program it has exactly one answer set, and that answer set equals
the Datalog model. :class:`AspProgram` runs the same clauses on clingo and rebuilds the
provenance the Datalog engine records: every clause gets a helper atom
``yappu_why_<i>(V1, …, Vn)`` over its variables. Each ground instance in the answer set
names the rule and the ground body that fired, so :meth:`Model.explain` prints the same
proof trees and :class:`AspRuleCoder` is a drop-in :class:`RuleCoder`.

What ASP adds on top is :meth:`AspRuleCoder.alternatives`. Excludes1 conflicts, which the
Datalog backend only flags for review, become a choice. For each conflicting pair, clingo
drops one code, ``#minimize`` keeps the drops to a minimum, and every optimal answer set is
returned as an alternative valid code set for the coder to choose from.

clingo is an optional dependency: ``pip install "yappu-fsm[asp]"``.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .coder import CodingResult, RuleCoder, _sequence
from .datalog import Fact, Literal, Model, Program, Rule, Var, _fmt

WHY = "yappu_why_"


def _require_clingo():
    try:
        import clingo  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError('the ASP backend needs clingo: pip install "yappu-fsm[asp]"') from e
    return clingo


def _lit(l: Literal) -> str:
    return str(l)                      # Datalog's printer already writes clingo syntax


def _rule_vars(r: Rule) -> List[Var]:
    vs = set(r.head.vars())
    for l in r.body:
        vs |= l.vars()
    return sorted(vs, key=lambda v: v.name)


def _ground(arg, binding):
    return binding[arg] if isinstance(arg, Var) else arg


def _sym_to_py(s) -> str:
    clingo = _require_clingo()
    if s.type == clingo.SymbolType.String:
        return s.string
    if s.type == clingo.SymbolType.Number:
        return str(s.number)
    if s.type == clingo.SymbolType.Function and not s.arguments:
        return s.name
    return str(s)


def _fact_text(f: Fact) -> str:
    return f"{f[0]}({', '.join(_fmt(a) for a in f[1:])})." if len(f) > 1 else f"{f[0]}."


class AspProgram:
    """The same clauses as a Datalog :class:`Program`, solved by clingo; ``run`` returns a
    :class:`Model` with provenance, so it can replace ``Program`` wherever ``run`` is used."""

    def __init__(self, program: Program):
        self.program = program
        self.rules: List[Rule] = list(program.rules)
        self._vars = [_rule_vars(r) for r in self.rules]
        lines = []
        for i, (r, vs) in enumerate(zip(self.rules, self._vars)):
            if r.source:
                lines.append(f"% {r.rid}: {r.source}")
            body = ", ".join(_lit(l) for l in r.body)
            lines.append(f"{_lit(r.head)} :- {body}." if body else f"{_lit(r.head)}.")
            why = f"{WHY}{i}({', '.join(v.name for v in vs)})" if vs else f"{WHY}{i}"
            lines.append(f"{why} :- {body}." if body else f"{why}.")
        self.text = "\n".join(lines)

    def to_text(self, provenance: bool = False) -> str:
        """The ASP program (by default without the provenance helper atoms)."""
        if provenance:
            return self.text
        return "\n".join(l for l in self.text.splitlines() if not l.startswith(WHY))

    def _solve(self, facts: Iterable[Fact], extra: str = "", opt: bool = False) -> List[List]:
        clingo = _require_clingo()
        ctl = clingo.Control(["--warn=none", "0"] + (["--opt-mode=optN"] if opt else []))
        ctl.add("base", [], self.text + "\n" + "\n".join(_fact_text(tuple(f)) for f in facts) + "\n" + extra)
        ctl.ground([("base", [])])
        models: List[Tuple[List, Optional[list]]] = []
        with ctl.solve(yield_=True) as handle:
            for m in handle:
                models.append((list(m.symbols(atoms=True)), list(m.cost), m.optimality_proven))
        if opt:                                   # keep only the proven-optimal models
            best = [x for x in models if x[2]]
            return [x[0] for x in best] if best else [models[-1][0]] if models else []
        return [x[0] for x in models]

    def _model(self, atoms, facts: Sequence[Fact]) -> Model:
        m = Model()
        given = [tuple(f) for f in facts]
        for f in given:
            if f not in m.facts:
                m.add(f, ("fact", ()))
        derived, whys = [], []
        for s in atoms:
            if s.name.startswith(WHY):
                whys.append((int(s.name[len(WHY):]), tuple(_sym_to_py(a) for a in s.arguments)))
            else:
                derived.append((s.name,) + tuple(_sym_to_py(a) for a in s.arguments))
        support: Dict[Fact, Tuple[str, tuple]] = {}
        for i, vals in sorted(whys):              # deterministic: earliest clause, then smallest binding
            r = self.rules[i]; b = dict(zip(self._vars[i], vals))
            head = (r.head.pred,) + tuple(_ground(a, b) for a in r.head.args)
            if head in support:
                continue
            used = []
            for l in r.body:
                g = (l.pred,) + tuple(_ground(a, b) for a in l.args)
                used.append(("not",) + g if l.negated else g)
            support[head] = (r.rid, tuple(used))
        for f in derived:
            if f not in m.facts:
                m.add(f, support.get(f, ("?", ())))
        return m

    def run(self, facts: Iterable[Fact]) -> Model:
        facts = [tuple(f) for f in facts]
        answer_sets = self._solve(facts)
        if len(answer_sets) != 1:                 # a stratified program has exactly one answer set
            raise RuntimeError(f"expected exactly one answer set, clingo found {len(answer_sets)}")
        return self._model(answer_sets[0], facts)


# Excludes1 as a choice: drop one code of each conflicting pair, as few codes as possible.
ALTERNATIVES = """
{ yappu_drop(P, A); yappu_drop(P, B) } :- conflict(P, A, B).
yappu_keep(P, C) :- final(P, C), not yappu_drop(P, C).
:- conflict(P, A, B), yappu_keep(P, A), yappu_keep(P, B).
:- yappu_drop(P, C), not final(P, C).
#minimize { 1, P, C : yappu_drop(P, C) }.
"""


class AspRuleCoder(RuleCoder):
    """:class:`RuleCoder` with clingo as the engine. ``code`` gives the same result as the
    Datalog backend (tested on every demo note); ``alternatives`` is ASP-only."""

    def __init__(self, linker, rules):
        super().__init__(linker, rules)
        self.asp = AspProgram(self.program)
        self.datalog_program = self.program
        self.program = self.asp                   # RuleCoder.code() only calls program.run(facts)

    def alternatives(self, note: str, setting: str = "outpatient") -> List[List[str]]:
        """Every minimal way to resolve the note's Excludes1 conflicts, as sequenced code sets.
        With no conflict this is a single set equal to ``code(note).codes``."""
        base = self.code(note, setting)
        if not base.conflicts:
            return [base.codes]
        facts = base.facts
        out = []
        for atoms in self.asp._solve(facts, ALTERNATIVES, opt=True):
            keep = {_sym_to_py(s.arguments[1]) for s in atoms if s.name == "yappu_keep"}
            codes = [c for c in base.codes if c in keep]
            if codes not in out:
                out.append(codes)
        return out
