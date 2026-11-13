"""Matthews Correlation Coefficient (binary and multiclass, Gorodkin R_K) and the
assertion / coding evaluation used by ``yappu eval``."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

MISSED = "MISSED"      # gold entity the system did not find
SPURIOUS = "NONE"      # system entity with no gold counterpart


def mcc(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Multiclass MCC (Gorodkin 2004); equals the usual binary MCC for 2 classes.

    Returns 0.0 when undefined (e.g. a single class only)."""
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    s = len(y_true)
    if s == 0:
        return 0.0
    c = sum(t == p for t, p in zip(y_true, y_pred))
    t_k, p_k = Counter(y_true), Counter(y_pred)
    labels = set(t_k) | set(p_k)
    num = c * s - sum(t_k[k] * p_k[k] for k in labels)
    den = math.sqrt((s * s - sum(p_k[k] ** 2 for k in labels)) * (s * s - sum(t_k[k] ** 2 for k in labels)))
    return 0.0 if den == 0 else num / den


def confusion(y_true, y_pred) -> Dict[Tuple[str, str], int]:
    return dict(Counter(zip(y_true, y_pred)))


def align(gold: List[Tuple[str, str, str]], pred: List[Tuple[str, str, str]]):
    """Pair gold and predicted (entity, assertion, laterality) by entity name, in order."""
    pool = list(pred)
    pairs = []
    for g in gold:
        idx = next((i for i, p in enumerate(pool) if p[0] == g[0]), None)
        if idx is None:
            pairs.append((g, None))
        else:
            pairs.append((g, pool.pop(idx)))
    pairs += [(None, p) for p in pool]
    return pairs


def evaluate(linker, gold_path: Union[str, Path]) -> dict:
    """Run the linker over a JSONL gold file and report MCC.

    * ``assertion_mcc``  – multiclass MCC over assertion labels (missed / spurious
      entities count as their own classes, so recall errors are penalised);
    * ``laterality_mcc`` – multiclass MCC over laterality labels for found entities;
    * ``codable_mcc``    – binary MCC of "should this mention produce a code?"
      (PRESENT / HISTORICAL / FAMILY_HISTORY vs NEGATED / UNCERTAIN / missed).
    """
    a_t, a_p, l_t, l_p, b_t, b_p = [], [], [], [], [], []
    errors = []
    for line in Path(gold_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold = [tuple(g) for g in row["gold"]]
        pred = [(f.entity, f.assertion, f.laterality) for f in linker.process(row["note"])]
        for g, p in align(gold, pred):
            gt = g[1] if g else SPURIOUS
            pt = p[1] if p else MISSED
            a_t.append(gt)
            a_p.append(pt)
            codable = lambda x: x in ("PRESENT", "HISTORICAL", "FAMILY_HISTORY")
            b_t.append(str(codable(gt)))
            b_p.append(str(codable(pt)))
            if g and p:
                l_t.append(g[2])
                l_p.append(p[2])
            if g != p:
                errors.append({"note": row["note"], "gold": g, "pred": p})
    return {
        "n_pairs": len(a_t),
        "assertion_mcc": round(mcc(a_t, a_p), 4),
        "laterality_mcc": round(mcc(l_t, l_p), 4),
        "codable_mcc": round(mcc(b_t, b_p), 4),
        "assertion_accuracy": round(sum(t == p for t, p in zip(a_t, a_p)) / max(1, len(a_t)), 4),
        "errors": errors,
    }
