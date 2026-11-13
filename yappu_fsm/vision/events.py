"""Tracks -> temporal facts -> controlled-English safety rules (Datalog) -> alerts with proofs.

Per frame, confirmed tracks become facts such as ``in_zone(7, restricted)`` and
``holds(7, cell-phone)``. A per-condition timer turns them into duration facts
(``in_zone_for(7, restricted, 2)`` once the condition has held for 2 s). A short gap
(missed detection, brief occlusion) does not reset the timer: that is the video
counterpart of a NegEx scope window. Rules are sentences, e.g.::

    INTRUSION: If a person T is in the restricted-zone for at least 2 seconds
               then a zone-intrusion alert is raised for the person T.

They compile to the same stratified Datalog engine as the ICD-10-CM rules, so every
alert carries a proof. An alert state machine (IDLE -> ACTIVE -> COOLING -> IDLE)
raises each alert once and clears it after a cool-down.

Rules fire on **positive evidence**. Writing "does not hold" is allowed, but in video the
absence of a detection may only mean occlusion or a dim frame, so such rules should be
paired with a duration condition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..rules.datalog import Literal, Program, Rule, Var
from .detector import COCO_CLASSES

SCENE = "scene"


# ============================================================================ zones

@dataclass
class Zone:
    name: str
    box: Tuple[float, float, float, float]      # x1, y1, x2, y2 as fractions of the frame (0..1)

    def contains(self, pt: Tuple[float, float], w: int, h: int) -> bool:
        x, y = pt[0] / w, pt[1] / h
        return self.box[0] <= x <= self.box[2] and self.box[1] <= y <= self.box[3]


def parse_zone(spec: str) -> Zone:
    """'restricted=0,0,0.4,1' -> Zone('restricted', (0, 0, 0.4, 1))"""
    name, coords = spec.split("=")
    vals = tuple(float(v) for v in coords.split(","))
    if len(vals) != 4 or not all(0 <= v <= 1 for v in vals):
        raise ValueError(f"zone '{spec}': expected name=x1,y1,x2,y2 with fractions in 0..1")
    return Zone(name.strip(), vals)


# ============================================================================ rule language

class VisionRuleError(ValueError):
    pass


@dataclass
class VisionRule:
    rid: str
    text: str
    clauses: List[Rule]
    durations: Set[Tuple[str, str, float]]       # (kind, arg, seconds) timers this rule needs
    crowds: Set[Tuple[str, int]]


_NUM = r"(\d+(?:\.\d+)?)"
_NAME = r"([a-z][a-z0-9]*(?:-[a-z0-9]+)*)"
_COND = [
    ("in_zone_for", re.compile(rf"^a person ([A-Z]) is in the {_NAME}-zone for at least {_NUM} seconds?$")),
    ("in_zone", re.compile(rf"^a person ([A-Z]) is in the {_NAME}-zone$")),
    ("holds_for", re.compile(rf"^the person ([A-Z]) holds an? {_NAME} for at least {_NUM} seconds?$")),
    ("holds", re.compile(rf"^the person ([A-Z]) holds an? {_NAME}$")),
    ("not_holds", re.compile(rf"^the person ([A-Z]) does not hold an? {_NAME}$")),
    ("visible_for", re.compile(rf"^a person ([A-Z]) is visible for at least {_NUM} seconds?$")),
    ("crowd", re.compile(rf"^at least (\d+) persons are in the {_NAME}-zone$")),
]
_CONSEQ = [
    ("alert_person", re.compile(rf"^an? {_NAME} alert is raised for the person ([A-Z])$")),
    ("alert_scene", re.compile(rf"^an? {_NAME} alert is raised$")),
]


def _object_name(n: str) -> str:
    label = n.replace("-", " ")
    if label not in COCO_CLASSES:
        raise VisionRuleError(f"'{n}' is not a COCO class the detector knows")
    return n


def parse_vision_rule(sentence: str, rid: str = "v1") -> VisionRule:
    s = sentence.strip()
    m = re.match(r"^If (.+) then (.+)\.$", s)
    if not m:
        raise VisionRuleError(f"{rid}: expected 'If ... then ... .'\n    {s}")
    conds = [c.strip() for c in re.split(r"\s+and\s+", m.group(1))]
    P = lambda name: Var(name)
    body: List[Literal] = []
    durations, crowds, bound = set(), set(), set()
    for c in conds:
        for kind, rx in _COND:
            g = rx.match(c)
            if g:
                break
        else:
            raise VisionRuleError(f"{rid}: condition not in the vision rule language: '{c}'")
        if kind == "in_zone_for":
            v, zone, sec = g.group(1), g.group(2), float(g.group(3))
            body.append(Literal("in_zone_for", (P(v), zone, sec)))
            durations.add(("in_zone", zone, sec))
            bound.add(v)
        elif kind == "in_zone":
            body.append(Literal("in_zone", (P(g.group(1)), g.group(2))))
            bound.add(g.group(1))
        elif kind == "holds_for":
            v, obj, sec = g.group(1), _object_name(g.group(2)), float(g.group(3))
            body.append(Literal("holds_for", (P(v), obj, sec)))
            durations.add(("holds", obj, sec))
            bound.add(v)
        elif kind == "holds":
            body.append(Literal("holds", (P(g.group(1)), _object_name(g.group(2)))))
            bound.add(g.group(1))
        elif kind == "not_holds":
            body.append(Literal("holds", (P(g.group(1)), _object_name(g.group(2))), True))
        elif kind == "visible_for":
            v, sec = g.group(1), float(g.group(2))
            body.append(Literal("visible_for", (P(v), sec)))
            durations.add(("visible", "", sec))
            bound.add(v)
        elif kind == "crowd":
            k, zone = int(g.group(1)), g.group(2)
            body.append(Literal("crowd", (zone, k)))
            crowds.add((zone, k))
    con = m.group(2).strip()
    for kind, rx in _CONSEQ:
        g = rx.match(con)
        if g:
            break
    else:
        raise VisionRuleError(f"{rid}: consequence not in the vision rule language: '{con}'")
    if kind == "alert_person":
        event, v = g.group(1), g.group(2)
        if v not in bound:
            raise VisionRuleError(f"{rid}: the person {v} must be introduced by a positive condition")
        head = Literal("alert", (P(v), event))
    else:
        head = Literal("alert", (SCENE, g.group(1)))
    return VisionRule(rid, s, [Rule(head, body, rid, s)], durations, crowds)


def parse_vision_rules(text: str) -> List[VisionRule]:
    out, n = [], 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        n += 1
        rid = f"v{n}"
        m = re.match(r"^([A-Za-z][\w.-]*):\s+(.*)$", line)
        if m:
            rid, line = m.group(1), m.group(2)
        out.append(parse_vision_rule(line, rid))
    return out


# ============================================================================ temporal facts

class Timer:
    """Tracks how long a condition has held, tolerating gaps shorter than ``grace``."""

    def __init__(self, grace: float):
        self.grace = grace
        self.start: Dict[tuple, float] = {}
        self.last_true: Dict[tuple, float] = {}

    def observe(self, key: tuple, true: bool, t: float):
        if true:
            if key not in self.start:
                self.start[key] = t
            self.last_true[key] = t
        elif key in self.last_true and t - self.last_true[key] > self.grace:
            self.start.pop(key, None)
            self.last_true.pop(key, None)

    def duration(self, key: tuple, t: float) -> float:
        if key not in self.start:
            return 0.0
        return self.last_true[key] - self.start[key] if t - self.last_true[key] <= self.grace else 0.0

    def active_keys(self) -> List[tuple]:
        return list(self.start)


@dataclass
class AlertEvent:
    kind: str            # "raised" | "cleared"
    subject: str         # track id or "scene"
    event: str
    t: float
    proof: List[str] = field(default_factory=list)


class AlertMachine:
    """Per (subject, event): IDLE -> ACTIVE (raise once) -> COOLING -> IDLE after ``cooldown`` s."""

    def __init__(self, cooldown: float = 3.0):
        self.cooldown = cooldown
        self.state: Dict[Tuple[str, str], Tuple[str, float]] = {}

    def step(self, derived: Set[Tuple[str, str]], t: float, proofs: Dict) -> List[AlertEvent]:
        out = []
        for key in derived:
            st = self.state.get(key, ("IDLE", t))[0]
            if st == "IDLE":
                out.append(AlertEvent("raised", key[0], key[1], t, proofs.get(key, [])))
            self.state[key] = ("ACTIVE", t)
        for key, (st, since) in list(self.state.items()):
            if key in derived:
                continue
            if st == "ACTIVE":
                self.state[key] = ("COOLING", t)
            elif st == "COOLING" and t - since >= self.cooldown:
                del self.state[key]
                out.append(AlertEvent("cleared", key[0], key[1], t))
        return out

    def active(self) -> List[Tuple[str, str]]:
        """Alerts whose condition holds right now (COOLING ones are still open but not shown)."""
        return [k for k, (st, _) in self.state.items() if st == "ACTIVE"]

    def open(self) -> List[Tuple[str, str]]:
        return list(self.state)


class EventEngine:
    def __init__(self, rules: Sequence[VisionRule], zones: Sequence[Zone], grace: float = 0.7,
                 cooldown: float = 3.0, hold_ioa: float = 0.5, include_lost: bool = True):
        self.rules, self.zones = list(rules), {z.name: z for z in zones}
        for r in self.rules:
            for kind, arg, _ in r.durations:
                if kind == "in_zone" and arg not in self.zones:
                    raise VisionRuleError(f"{r.rid}: no zone named '{arg}' (have: {sorted(self.zones)})")
            for zone, _ in r.crowds:
                if zone not in self.zones:
                    raise VisionRuleError(f"{r.rid}: no zone named '{zone}'")
        self.program = Program([c for r in self.rules for c in r.clauses])
        self.program.strata()
        self.durations = {d for r in self.rules for d in r.durations}
        self.crowds = {c for r in self.rules for c in r.crowds}
        self.rules_by_id = {r.rid: r.clauses[0] for r in self.rules}
        self.timer = Timer(grace)
        self.alerts = AlertMachine(cooldown)
        self.hold_ioa, self.include_lost = hold_ioa, include_lost
        self.last_facts: List[tuple] = []

    @staticmethod
    def _ioa(inner, outer) -> float:
        x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
        x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return inter / area

    def facts_for(self, tracks, t: float, frame_wh: Tuple[int, int]) -> List[tuple]:
        w, h = frame_wh
        persons = [tr for tr in tracks if COCO_CLASSES[tr.cls] == "person"]
        objects = [tr for tr in tracks if COCO_CLASSES[tr.cls] != "person"]
        facts: List[tuple] = []
        seen_now = set()
        for p in persons:
            pid = str(p.tid)
            cx, cy = (p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2
            facts.append(("person", pid))
            self.timer.observe(("visible", pid, ""), True, t)
            seen_now.add(("visible", pid, ""))
            for z in self.zones.values():
                inside = z.contains((cx, cy), w, h)
                if inside:
                    facts.append(("in_zone", pid, z.name))
                self.timer.observe(("in_zone", pid, z.name), inside, t)
                seen_now.add(("in_zone", pid, z.name))
            held = set()
            for o in objects:
                if self._ioa(o.box, p.box) >= self.hold_ioa:
                    held.add(COCO_CLASSES[o.cls].replace(" ", "-"))
            for name in held:
                facts.append(("holds", pid, name))
            for kind, obj, _ in self.durations:
                if kind == "holds":
                    self.timer.observe(("holds", pid, obj), obj in held, t)
                    seen_now.add(("holds", pid, obj))
        # timers for tracks not seen this frame keep running only within the grace period
        for key in self.timer.active_keys():
            if key not in seen_now:
                self.timer.observe(key, False, t)
        # duration facts
        for kind, pid, arg in self.timer.active_keys():
            d = self.timer.duration((kind, pid, arg), t)
            for dk, darg, sec in self.durations:
                if dk == kind and (darg == arg) and d >= sec:
                    facts.append(("visible_for", pid, sec) if kind == "visible" else (f"{kind}_for", pid, arg, sec))
        for zone, k in self.crowds:
            n = sum(1 for f in facts if f[0] == "in_zone" and f[2] == zone)
            if n >= k:
                facts.append(("crowd", zone, k))
        self.last_facts = facts
        return facts

    def step(self, tracks, t: float, frame_wh: Tuple[int, int]) -> List[AlertEvent]:
        m = self.program.run(self.facts_for(tracks, t, frame_wh))
        derived = {(f[1], f[2]) for f in m.query("alert")}
        proofs = {(f[1], f[2]): m.explain(f, self.rules_by_id) for f in m.query("alert")}
        return self.alerts.step(derived, t, proofs)
