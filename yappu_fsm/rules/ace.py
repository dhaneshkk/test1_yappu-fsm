"""A controlled-English rule language for ICD-10-CM coding, compiled to Datalog.

The sentences follow the conventions of Attempto Controlled English (ACE): a
fixed set of function words, user-defined content words (here: lexicon concepts
written with hyphens, e.g. ``type-2-diabetes``), simple present tense, a
deterministic reading shown back as a bracketed paraphrase, and a precise error
when a sentence is outside the language.  It is a small *coding-rule subset*
in the spirit of ACE, parsed by this module; it does not call the Attempto
Parsing Engine.

Grammar (one rule per sentence)::

    rule     := "If" cond ("and" cond)* "then" conseq ("and" conseq)* "."
    cond     := subj "has" term
              | subj "does not have" term
              | subj "has" ("a"|"an") "uncertain" (CONCEPT | "diagnosis" VAR)
              | subj "has a history of" term
              | "a relative of the patient has" term
              | "the encounter is" ("a"|"an") ("outpatient-encounter" | "inpatient-encounter")
              | "the" (CONCEPT|VAR) "is on the" ("left"|"right") "side"
              | "the" (CONCEPT|VAR) "is on both sides"
              | "the code" (CODE|VAR) "is" ["not"] "assigned"
              | "a code" VAR "is assigned"                      (introduces a code variable)
              | "the code" (CODE|VAR) "is a" KIND "code"       KIND: diagnosis | symptom | factor | external
    conseq   := "the code" CODE "is" ["not"] "assigned"
              | "the candidate code of" VAR "is assigned"
              | "the code" (CODE|VAR) "precedes the code" (CODE|VAR)
              | "the codes" CODE "and" CODE "conflict"
    subj     := ("a"|"the") "patient"
    term     := ("a"|"an") CONCEPT | ("a"|"an") "diagnosis" VAR | VAR | "the" CONCEPT

Datalog vocabulary produced: patient/1 (implicit in every rule), has/2, uncertain/2, history/2, family/2, setting/2,
side/3, candidate/3, kind/3, code/2, blocked/2, precedes/3, conflict/3 and the built-in
``final(P, C) :- code(P, C), not blocked(P, C).``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from .datalog import Literal, Program, Rule, Var

CODE_RE = re.compile(r"^[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$")
VAR_RE = re.compile(r"^[A-Z][0-9]?$")
CONCEPT_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
P = Var("P")
SETTINGS = {"outpatient-encounter": "outpatient", "inpatient-encounter": "inpatient"}
SIDES = {"left": "left", "right": "right", "both": "both"}
KINDS = ("diagnosis", "symptom", "factor", "external")
FINAL_RULE = Rule(Literal("final", (P, Var("C"))), [Literal("code", (P, Var("C"))), Literal("blocked", (P, Var("C")), True)],
                  "final", "A code is final if it is assigned and not blocked.")


class AceError(ValueError):
    def __init__(self, sentence: str, pos: int, expected: str, found: str, line: Optional[int] = None):
        self.sentence, self.pos, self.expected, self.found, self.line = sentence, pos, expected, found, line
        where = f"line {line}: " if line else ""
        super().__init__(f"{where}not in the coding-rule language: expected {expected} at word {pos + 1}, "
                         f"found '{found}'\n    {sentence}")


@dataclass
class Atom:
    kind: str                  # has | not_has | uncertain | history | family | setting | side | code | not_code
    args: Tuple[object, ...]

    def paraphrase(self) -> str:
        a = [_say(x) for x in self.args] + ["", ""]
        return {
            "has": f"the patient has {a[0]}",
            "not_has": f"it is false that the patient has {a[0]}",
            "uncertain": f"the patient has an uncertain {a[0]}",
            "history": f"the patient has a history of {a[0]}",
            "family": f"a relative of the patient has {a[0]}",
            "setting": f"the encounter is {a[0]}",
            "side": f"{a[0]} is on side {a[1]}",
            "code": f"code {a[0]} is assigned",
            "kind": f"code {a[0]} is a {a[1]} code",
            "not_code": f"it is false that code {a[0]} is assigned",
            "assign": f"code {a[0]} is assigned",
            "assign_candidate": f"the candidate code of {a[0]} is assigned",
            "block": f"code {a[0]} is blocked",
            "precedes": f"code {a[0]} is sequenced before code {a[1]}",
            "conflict": f"codes {a[0]} and {a[1]} conflict",
        }[self.kind]


def _say(x) -> str:
    return x.name if isinstance(x, Var) else str(x).replace("-", " ")


@dataclass
class AceRule:
    rid: str
    text: str
    conds: List[Atom]
    conseqs: List[Atom]

    def paraphrase(self) -> str:
        c = " AND ".join("{ " + a.paraphrase() + " }" for a in self.conds)
        t = " AND ".join("{ " + a.paraphrase() + " }" for a in self.conseqs)
        return f"IF {c} THEN {t}"

    def to_datalog(self) -> List[Rule]:
        body: List[Literal] = [Literal("patient", (P,))]   # binds P even when all conditions are negative
        for a in self.conds:
            k, args = a.kind, a.args
            if k == "not_has":
                body.append(Literal("has", (P,) + args, True))
            elif k == "not_code":
                body.append(Literal("code", (P,) + args, True))
            else:
                body.append(Literal(k, (P,) + args))
        rules = []
        for i, c in enumerate(self.conseqs):
            rid = self.rid if len(self.conseqs) == 1 else f"{self.rid}.{i + 1}"
            if c.kind == "assign":
                rules.append(Rule(Literal("code", (P,) + c.args), list(body), rid, self.text))
            elif c.kind == "assign_candidate":
                cv = Var("C_" + c.args[0].name)
                rules.append(Rule(Literal("code", (P, cv)), body + [Literal("candidate", (P, c.args[0], cv))], rid, self.text))
            elif c.kind == "block":
                rules.append(Rule(Literal("blocked", (P,) + c.args), list(body), rid, self.text))
            elif c.kind in ("precedes", "conflict"):
                rules.append(Rule(Literal(c.kind, (P,) + c.args), list(body), rid, self.text))
        return rules


class _Parser:
    def __init__(self, sentence: str, concepts: Optional[Set[str]], line: Optional[int]):
        self.s = sentence.strip()
        self.toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*[A-Za-z0-9]|[A-Za-z0-9]|[.,]", self.s)
        if self.toks and self.toks[-1] != "." and self.toks[-1].endswith("."):
            self.toks[-1] = self.toks[-1][:-1]
            self.toks.append(".")
        self.i, self.concepts, self.line = 0, concepts, line
        self.vars: Set[str] = set()

    # helpers -------------------------------------------------------------
    def peek(self, k=0) -> str:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else "<end>"

    def fail(self, expected: str):
        raise AceError(self.s, self.i, expected, self.peek(), self.line)

    def word(self, *options: str) -> str:
        t = self.peek()
        if t.lower() in options:
            self.i += 1
            return t.lower()
        self.fail(" or ".join(f"'{o}'" for o in options))

    def phrase(self, text: str):
        for w in text.split():
            self.word(w)

    def at(self, text: str) -> bool:
        ws = text.split()
        return all(self.peek(k).lower() == w for k, w in enumerate(ws))

    def code(self) -> str:
        t = self.peek()
        if not CODE_RE.match(t):
            self.fail("an ICD-10-CM code such as E11.22")
        self.i += 1
        return t

    def code_or_var(self):
        t = self.peek()
        if CODE_RE.match(t):
            return self.code()
        if VAR_RE.match(t):
            return self.var()
        self.fail("an ICD-10-CM code or a code variable")

    def concept(self) -> str:
        t = self.peek()
        if not CONCEPT_RE.match(t) or t in ("diagnosis", "uncertain", "history", "patient"):
            self.fail("a concept word such as type-2-diabetes")
        if self.concepts is not None and t.replace("-", " ") not in self.concepts:
            raise AceError(self.s, self.i, "a concept from the lexicon", t, self.line)
        self.i += 1
        return t

    def var(self, introduce=False) -> Var:
        t = self.peek()
        if not VAR_RE.match(t):
            self.fail("a variable such as X")
        if not introduce and t not in self.vars:
            raise AceError(self.s, self.i, "a variable introduced earlier ('a diagnosis X')", t, self.line)
        self.vars.add(t)
        self.i += 1
        return Var(t)

    # grammar -------------------------------------------------------------
    def term(self):
        t = self.peek().lower()
        if t in ("a", "an"):
            self.i += 1
            if self.peek().lower() == "diagnosis":
                self.i += 1
                return self.var(introduce=True)
            return self.concept()
        if t == "the":
            self.i += 1
            return self.concept()
        if VAR_RE.match(self.peek()):
            return self.var()
        self.fail("'a', 'an', 'the' or a variable")

    def cond(self) -> Atom:
        t = self.peek().lower()
        if t in ("a", "the") and self.peek(1).lower() == "patient":
            self.i += 2
            if self.at("does not have"):
                self.phrase("does not have")
                return Atom("not_has", (self.term(),))
            self.word("has")
            if self.at("a history of"):
                self.phrase("a history of")
                return Atom("history", (self.term(),))
            if self.peek().lower() in ("a", "an") and self.peek(1).lower() == "uncertain":
                self.i += 2
                if self.peek().lower() == "diagnosis":
                    self.i += 1
                    return Atom("uncertain", (self.var(introduce=True),))
                return Atom("uncertain", (self.concept(),))
            return Atom("has", (self.term(),))
        if self.at("a relative of the patient has"):
            self.phrase("a relative of the patient has")
            return Atom("family", (self.term(),))
        if self.at("the encounter is"):
            self.phrase("the encounter is")
            self.word("a", "an")
            s = self.word(*SETTINGS)
            return Atom("setting", (SETTINGS[s],))
        if self.at("a code"):
            self.phrase("a code")
            v = self.var(introduce=True)
            self.phrase("is assigned")
            return Atom("code", (v,))
        if self.at("the code"):
            self.phrase("the code")
            c = self.code_or_var()
            self.word("is")
            if self.peek().lower() == "a":
                self.i += 1
                k = self.word(*KINDS)
                self.word("code")
                return Atom("kind", (c, k))
            neg = self.peek().lower() == "not"
            if neg:
                self.i += 1
            self.word("assigned")
            return Atom("not_code" if neg else "code", (c,))
        if t == "the":
            self.i += 1
            subj = self.var() if VAR_RE.match(self.peek()) else self.concept()
            self.phrase("is on")
            if self.peek().lower() == "both":
                self.phrase("both sides")
                return Atom("side", (subj, "both"))
            self.word("the")
            s = self.word("left", "right")
            self.word("side")
            return Atom("side", (subj, s))
        self.fail("a condition ('the patient has ...', 'the code ... is assigned', 'the encounter is ...')")

    def conseq(self) -> Atom:
        if self.at("the candidate code of"):
            self.phrase("the candidate code of")
            v = self.var()
            self.phrase("is assigned")
            return Atom("assign_candidate", (v,))
        if self.at("the codes"):
            self.phrase("the codes")
            a = self.code()
            self.word("and")
            b = self.code()
            self.word("conflict")
            return Atom("conflict", (a, b))
        self.phrase("the code")
        c = self.code_or_var()
        if self.peek().lower() == "precedes":
            self.phrase("precedes the code")
            return Atom("precedes", (c, self.code_or_var()))
        if isinstance(c, Var):
            self.fail("'precedes' (only sequencing may use a code variable)")
        self.word("is")
        if self.peek().lower() == "not":
            self.i += 1
            self.word("assigned")
            return Atom("block", (c,))
        self.word("assigned")
        return Atom("assign", (c,))

    def rule(self, rid: str) -> AceRule:
        self.word("if")
        conds = [self.cond()]
        while self.peek().lower() == "and":
            self.i += 1
            conds.append(self.cond())
        self.word("then")
        conseqs = [self.conseq()]
        while self.peek().lower() == "and":
            self.i += 1
            conseqs.append(self.conseq())
        self.word(".")
        if self.i != len(self.toks):
            self.fail("end of sentence")
        return AceRule(rid, self.s, conds, conseqs)


def parse_rule(sentence: str, rid: str = "r1", concepts: Optional[Iterable[str]] = None, line: Optional[int] = None) -> AceRule:
    return _Parser(sentence, set(concepts) if concepts is not None else None, line).rule(rid)


def parse_rules(text: str, concepts: Optional[Iterable[str]] = None) -> List[AceRule]:
    """Parse a rules file: one sentence per line; '#' starts a comment; an optional
    'ID:' prefix names the rule (otherwise r1, r2, ...)."""
    cs = set(concepts) if concepts is not None else None
    out, n = [], 0
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        n += 1
        rid = f"r{n}"
        m = re.match(r"^([A-Za-z][\w.-]*):\s+(.*)$", line)
        if m:
            rid, line = m.group(1), m.group(2)
        out.append(parse_rule(line, rid, cs, ln))
    ids = [r.rid for r in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate rule ids: {sorted(dup)}")
    return out


def compile_rules(rules: Sequence[AceRule]) -> Program:
    prog = Program()
    for r in rules:
        for d in r.to_datalog():
            prog.add(d)
    prog.add(Rule(FINAL_RULE.head, list(FINAL_RULE.body), FINAL_RULE.rid, FINAL_RULE.source))
    prog.strata()  # raises DatalogError if the rules are not stratifiable
    return prog
