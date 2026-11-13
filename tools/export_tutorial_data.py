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
    "Denies SOB, history of MI. Father had colon cancer.",
    "Probable pneumonia, start antibiotics. Rt CTS confirmed on EMG.",
    "Bilateral hip pain worse on stairs. No change in hypertension since last visit.",
    "No known drug allergies, takes lisinopril for longstanding hypertension.",
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
        for e in linker.match_entities(toks):
            a = linker._assertion(cl, trig, e["c0"], e["c1"])
            lat, dist = linker._bind_laterality(e["t0"], e["t1"], lats)
            ents.append({"raw": e["raw"], "entity": e["entry"]["canonical"], "match": e["match_type"],
                         "score": e["score"], "c0": e["c0"], "c1": e["c1"], "assertion": a,
                         "laterality": lat, "dist": dist})
        clauses.append({"text": clause, "triggers": [{"c0": a, "c1": b, "state": s} for a, b, s in trig],
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


lex = VenbaLexicon.sample()
toks, vocab = tokenization()
data = {
    "traces": [trace(ClinicalLinker(LEX), n) for n in NOTES],
    "tokenization": toks, "vocab_size": vocab,
    "confusion": confusion(),
    "lexicon": [{"w": e.word, "a": e.asai, "c": e.category, "f": e.final_ok, "e": e.ethukai, "m": e.monai}
                for e in lex.entries],
    "icd_lexicon_size": len(LEX),
}
out = ROOT / "docs" / "tutorial_data.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print(out, out.stat().st_size, "bytes")
