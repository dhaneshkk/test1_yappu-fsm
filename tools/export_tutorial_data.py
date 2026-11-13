"""Export real traces from the package for docs/tutorial.html."""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yappu_fsm import data_path  # noqa: E402
from yappu_fsm.clinical import ClinicalLinker, evaluate, load_lexicon  # noqa: E402
from yappu_fsm.clinical.metrics import align  # noqa: E402
from yappu_fsm.lexicon import VenbaLexicon  # noqa: E402

LEX = load_lexicon(data_path("icd10cm_example_lexicon.json"))
NOTES = [
    "Pt has lt knee osteoarthritus and mild NPDR OD. Denies SOB, but has uncontrolled HTN. R/O CAP.",
    "No fever aside from mild cough. Return if headache worsens.",
    "Pneumonia in the left lower lobe was ruled out. No history of MI.",
    "Denies SOB, history of MI. Father had colon cancer.",
    "Probable pneumonia, start antibiotics. Rt CTS confirmed on EMG.",
    "No change in hypertension since last visit. Bilateral hip pain worse on stairs.",
]
RULE_NOTES = [
    "Pt with T2DM and CKD. Also HTN. Probable CAP. Cough for three days.",
    "Pneumonia confirmed on imaging. Cough and fever for three days.",
    "T1DM noted; also T2DM per old records.",
    "Cough for three days. Family history of HTN. Probable CAP. Pt with T2DM and CKD.",
    "History of CVA. Headache. Left knee osteoarthritis.",
]


def trace(linker, note):
    expanded = linker.expand_abbreviations(note)
    clauses = []
    for clause in linker.split_clauses(expanded):
        cl = clause.lower()
        toks = linker._tokens(cl)
        if not toks:
            continue
        trig = linker._triggers(cl)
        lats = linker._laterality_mentions(cl, toks)
        lat_spans = [{"label": m["label"], "c0": toks[m["t0"]][1], "c1": toks[m["t1"]][2]} for m in lats]
        ents = []
        post = linker._post_hits(cl)
        for e in linker.match_entities(toks):
            a = linker._assertion(cl, trig, e["c0"], e["c1"])
            scope = None
            pre = [t for t in trig if t[1] <= e["c0"]]
            if pre:
                t = max(pre, key=lambda t: t[1])
                n = linker._ntok(cl[t[1]:e["c0"]])
                scope = {"dir": "pre", "c0": t[0], "c1": t[1], "state": t[2], "dist": n,
                         "conj": linker._conj_between(cl, t[1], e["c0"]), "used": n <= linker.max_scope and not linker._conj_between(cl, t[1], e["c0"])}
            po = [t for t in post if t[0] >= e["c1"]]
            if po:
                t = min(po, key=lambda t: t[0])
                n = linker._ntok(cl[e["c1"]:t[0]])
                ok = n <= linker.max_scope and not linker._conj_between(cl, e["c1"], t[0])
                if ok or scope is None or not scope["used"]:
                    scope = {"dir": "post", "c0": t[0], "c1": t[1], "state": t[2], "dist": n,
                             "conj": linker._conj_between(cl, e["c1"], t[0]), "used": ok}
            lat, dist = linker._bind_laterality(e["t0"], e["t1"], lats)
            ents.append({"raw": e["raw"], "entity": e["entry"]["canonical"], "match": e["match_type"],
                         "score": e["score"], "c0": e["c0"], "c1": e["c1"], "assertion": a,
                         "laterality": lat, "dist": dist, "scope": scope})
        conj = [{"c0": m.start(), "c1": m.end()} for p in linker.CONJ for m in re.finditer(p, cl)]
        clauses.append({"text": clause, "conj": conj, "post": [{"c0": a, "c1": b, "state": s} for a, b, s in post],
                        "triggers": [{"c0": a, "c1": b, "state": s} for a, b, s in trig],
                        "lats": lat_spans, "entities": ents})
    findings = {s: [f.to_dict() for f in ClinicalLinker(LEX, setting=s).process(note)]
                for s in ("outpatient", "inpatient")}
    return {"note": note, "expanded": expanded, "clauses": clauses, "findings": findings}


def tokenization():
    from yappu_fsm.decoding import token_bytes_table
    from yappu_fsm.demo import build_tiny_model
    from yappu_fsm.lexicon import SAMPLE_WORDS
    codes = [c for v in LEX.values() for vv in v.values() for c in ([vv] if isinstance(vv, str) else vv)]
    tok, _ = build_tiny_model([" ".join(SAMPLE_WORDS)] * 4 + ["CODES: " + "; ".join(codes)] * 4)
    table = token_bytes_table(tok)
    out = []
    for text in ["வண்ண நிலவொளி", "கூடினால்", "CODES: E11.3291; I10"]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out.append({"text": text, "tokens": [{"id": i, "hex": table[i].hex(" "),
                    "text": table[i].decode("utf-8", errors="replace")} for i in ids],
                    "utf8": text.encode().hex(" ")})
    return out, len(tok)


def confusion():
    r = evaluate(ClinicalLinker(LEX), data_path("gold_assertions.jsonl"))
    t, p = [], []
    for line in pathlib.Path(data_path("gold_assertions.jsonl")).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        pred = [(f.entity, f.assertion, f.laterality) for f in ClinicalLinker(LEX).process(row["note"])]
        for g, q in align([tuple(x) for x in row["gold"]], pred):
            t.append(g[1] if g else "NONE")
            p.append(q[1] if q else "MISSED")
    labels = sorted(set(t) | set(p), key=["PRESENT", "NEGATED", "UNCERTAIN", "HISTORICAL", "FAMILY_HISTORY", "MISSED", "NONE"].index)
    m = [[sum(1 for a, b in zip(t, p) if a == x and b == y) for y in labels] for x in labels]
    return {"labels": labels, "matrix": m, "summary": {k: v for k, v in r.items() if k != "errors"},
            "errors": r["errors"]}


def rules_data():
    from yappu_fsm.rules import RuleCoder
    from yappu_fsm.rules.datalog import Var
    coder = RuleCoder(ClinicalLinker(LEX), data_path("icd10cm_rules.ace"))
    strata = coder.program.strata()
    level = {id(r): i for i, st in enumerate(strata) for r in st}
    enc = lambda a: {"v": a.name} if isinstance(a, Var) else a
    lit = lambda l: {"pred": l.pred, "args": [enc(a) for a in l.args], "neg": l.negated}
    clauses = [{"rid": r.rid, "base": r.rid.split(".")[0], "head": lit(r.head), "body": [lit(b) for b in r.body],
                "stratum": level[id(r)], "text": str(r)} for r in coder.program.rules]
    rules = [{"rid": r.rid, "text": r.text, "paraphrase": r.paraphrase()} for r in coder.rules]
    notes = []
    for n in RULE_NOTES:
        entry = {"note": n, "findings": [], "facts": {}, "expected": {}}
        for setting in ("outpatient", "inpatient"):
            res = coder.code(n, setting)
            entry["facts"][setting] = [list(f) for f in res.facts]
            entry["expected"][setting] = {"codes": res.codes, "blocked": sorted(res.blocked),
                                          "conflicts": [[a, b] for a, b, _ in res.conflicts]}
            entry["findings"] = [{"entity": f.entity, "assertion": f.assertion, "codes": f.codes} for f in res.findings]
        notes.append(entry)
    # expected results with each rule disabled (for the JS port test)
    ablation = {}
    from yappu_fsm.rules import RuleCoder as RC
    for r in coder.rules:
        sub = RC(ClinicalLinker(LEX), [x for x in coder.rules if x.rid != r.rid])
        ablation[r.rid] = {n: sub.code(n, "inpatient").codes for n in RULE_NOTES}
    return {"rules": rules, "clauses": clauses, "notes": notes, "ablation": ablation,
            "datalog": coder.program.to_text()}


def gold_summary():
    out = []
    for name, label in [("gold_assertions.jsonl", "original synthetic set"), ("gold_negex.jsonl", "NegEx tuning set"),
                        ("gold_negex_heldout.jsonl", "NegEx second set")]:
        r = evaluate(ClinicalLinker(LEX), data_path(name))
        out.append({"name": name, "label": label, "assertion_mcc": r["assertion_mcc"], "codable_mcc": r["codable_mcc"],
                    "n": r["n_pairs"]})
    return out


def main():
    lex = VenbaLexicon.sample()
    toks, vocab = tokenization()
    data = {
        "traces": [trace(ClinicalLinker(LEX), n) for n in NOTES],
        "tokenization": toks, "vocab_size": vocab,
        "confusion": confusion(),
        "lexicon": [{"w": e.word, "a": e.asai, "c": e.category, "f": e.final_ok, "e": e.ethukai, "m": e.monai}
                    for e in lex.entries],
        "icd_lexicon_size": len(LEX),
        "rules": rules_data(),
        "gold_sets": gold_summary(),
        # measured in this repo's sandbox (Python 3.13, CPU), see README "Performance"
        "heldout_before_after": {"before": {"assertion_mcc": 0.619, "codable_mcc": 0.539},
                                 "after_first_run": {"assertion_mcc": 0.861, "codable_mcc": 0.811}},
        "bench": {"sizes": [32, 1000, 5000, 20000, 100000], "full_scan_ms": [3.14, 125.49, 691.32, 2434.94, None],
                  "indexed_ms": [0.19, 0.19, None, 0.35, 0.42], "rules_ms": 0.30, "linker_ms": 0.14},
    }
    out = ROOT / "docs" / "tutorial_data.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
