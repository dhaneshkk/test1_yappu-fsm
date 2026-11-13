"""A small Datalog engine: stratified negation, semi-naive evaluation, provenance.

Terms are plain strings; variables are :class:`Var` objects.  Facts are tuples
``(predicate, arg1, arg2, ...)``.  Every derived fact remembers the rule and the
ground body facts that first produced it, so :meth:`Program.explain` can print
a proof tree.

Datalog (Horn clauses without function symbols, with stratified negation) always
terminates and runs in polynomial time in the number of facts – unlike general
first-order theorem proving, which is only semi-decidable.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Fact = Tuple[str, ...]


@dataclass(frozen=True)
class Var:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Literal:
    pred: str
    args: Tuple[object, ...]
    negated: bool = False

    def vars(self) -> Set[Var]:
        return {a for a in self.args if isinstance(a, Var)}

    def __str__(self) -> str:
        s = f"{self.pred}({', '.join(_fmt(a) for a in self.args)})"
        return ("not " + s) if self.negated else s


@dataclass
class Rule:
    head: Literal
    body: List[Literal]
    rid: str = ""
    source: str = ""          # original ACE sentence, if any

    def __str__(self) -> str:
        return f"{self.head} :- {', '.join(map(str, self.body))}." if self.body else f"{self.head}."


def _fmt(a) -> str:
    if isinstance(a, Var):
        return a.name
    s = str(a)
    return s if re.fullmatch(r"[a-z][a-z0-9_]*", s) else '"' + s.replace('"', '\\"') + '"'


class DatalogError(ValueError):
    pass


class Program:
    def __init__(self, rules: Iterable[Rule] = ()):
        self.rules: List[Rule] = []
        for r in rules:
            self.add(r)

    def add(self, rule: Rule):
        if rule.head.negated:
            raise DatalogError(f"{rule.rid}: negated head")
        positive = set().union(*[l.vars() for l in rule.body if not l.negated]) if rule.body else set()
        unsafe = (rule.head.vars() | set().union(*[l.vars() for l in rule.body if l.negated] or [set()])) - positive
        if unsafe:
            raise DatalogError(f"{rule.rid}: unsafe variable(s) {sorted(v.name for v in unsafe)} "
                               f"must appear in a positive condition")
        # evaluate positive literals first, negations last (safe rules make this sound)
        rule.body = [l for l in rule.body if not l.negated] + [l for l in rule.body if l.negated]
        self.rules.append(rule)

    # ------------------------------------------------------------ strata
    def strata(self) -> List[List[Rule]]:
        preds = {r.head.pred for r in self.rules} | {l.pred for r in self.rules for l in r.body}
        level = {p: 0 for p in preds}
        n = len(preds)
        changed = True
        while changed:
            changed = False
            for r in self.rules:
                for l in r.body:
                    need = level[l.pred] + (1 if l.negated else 0)
                    if level[r.head.pred] < need:
                        level[r.head.pred] = need
                        changed = True
                        if need > n:
                            raise DatalogError(f"not stratifiable: '{r.head.pred}' depends negatively on itself "
                                               f"(via rule {r.rid})")
        out: Dict[int, List[Rule]] = defaultdict(list)
        for r in self.rules:
            out[level[r.head.pred]].append(r)
        return [out[k] for k in sorted(out)]

    # ------------------------------------------------------------ evaluation
    def run(self, facts: Iterable[Fact]) -> "Model":
        model = Model()
        for f in facts:
            model.add(tuple(f), ("fact", ()))
        for stratum in self.strata():
            self._fixpoint(stratum, model)
        return model

    def _fixpoint(self, rules: List[Rule], m: "Model"):
        delta: Set[Fact] = set(m.facts)          # first round: everything is new
        while delta:
            new: Set[Fact] = set()
            delta_by_pred = defaultdict(list)
            for f in delta:
                delta_by_pred[f[0]].append(f)
            for r in rules:
                pos = [i for i, l in enumerate(r.body) if not l.negated]
                # semi-naive: at least one positive literal must bind to a delta fact
                for di in pos:
                    if r.body[di].pred not in delta_by_pred:
                        continue
                    for binding, used in self._match(r.body, 0, {}, [], m, di, delta_by_pred):
                        head = (r.head.pred,) + tuple(binding[a] if isinstance(a, Var) else a for a in r.head.args)
                        if head not in m.facts and head not in new:
                            new.add(head)
                            m.add(head, (r.rid, tuple(used)))
                if not pos and r.body == []:
                    head = (r.head.pred,) + tuple(r.head.args)
                    if head not in m.facts:
                        new.add(head)
                        m.add(head, (r.rid, ()))
            delta = new

    def _match(self, body, i, binding, used, m, delta_i, delta_by_pred):
        """Enumerate bindings for body[i:]; literal ``delta_i`` reads only new facts."""
        if i == len(body):
            yield binding, used
            return
        lit = body[i]
        if lit.negated:  # all variables are bound here (positives come first)
            ground = (lit.pred,) + tuple(binding[a] if isinstance(a, Var) else a for a in lit.args)
            if ground not in m.facts:
                yield from self._match(body, i + 1, binding, used + [("not",) + ground], m, delta_i, delta_by_pred)
            return
        source = delta_by_pred.get(lit.pred, []) if i == delta_i else m.by_pred.get(lit.pred, [])
        for f in source:
            if len(f) != len(lit.args) + 1:
                continue
            b = dict(binding)
            for a, v in zip(lit.args, f[1:]):
                if isinstance(a, Var):
                    if b.setdefault(a, v) != v:
                        break
                elif a != v:
                    break
            else:
                yield from self._match(body, i + 1, b, used + [f], m, delta_i, delta_by_pred)

    def to_text(self) -> str:
        lines = []
        for r in self.rules:
            if r.source:
                lines.append(f"% {r.rid}: {r.source}")
            lines.append(str(r))
        return "\n".join(lines)


class Model:
    def __init__(self):
        self.facts: Set[Fact] = set()
        self.by_pred: Dict[str, List[Fact]] = defaultdict(list)
        self.why: Dict[Fact, Tuple[str, tuple]] = {}

    def add(self, f: Fact, why):
        self.facts.add(f)
        self.by_pred[f[0]].append(f)
        self.why[f] = why

    def query(self, pred: str) -> List[Fact]:
        return list(self.by_pred.get(pred, []))

    def explain(self, f: Fact, rules_by_id: Optional[Dict[str, Rule]] = None, depth: int = 0, seen=None,
                hide: Sequence[str] = ()) -> List[str]:
        seen = set() if seen is None else seen
        pad = "  " * depth
        if f[0] == "not":
            return [f"{pad}not {f[1]}({', '.join(map(str, f[2:]))})  [absent]"]
        rid, used = self.why.get(f, ("?", ()))
        label = "given" if rid == "fact" else rid
        src = ""
        if rules_by_id and rid in rules_by_id and rules_by_id[rid].source:
            src = f"  « {rules_by_id[rid].source} »"
        out = [f"{pad}{f[0]}({', '.join(map(str, f[1:]))})  [{label}]{src}"]
        if f in seen:
            return out
        seen.add(f)
        for u in used:
            if u[0] in hide:
                continue
            out += self.explain(u, rules_by_id, depth + 1, seen, hide)
        return out
