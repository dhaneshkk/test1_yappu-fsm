"""Command-line interface: ``yappu <command> ...``"""

from __future__ import annotations

import argparse
import json
import sys

from . import data_path
from .prosody import SEER, eetrucheer_vaipadu, format_venba, validate_nerisai_venba
from .tamil import encode, ethukai_key, monai_key, split_display


def cmd_asai(a):
    for w in a.words:
        code = encode(w)
        vp, cat = SEER.get(code, (eetrucheer_vaipadu(w), "Orasai/final" if eetrucheer_vaipadu(w) else None))
        print(f"{w:<16} {split_display(w):<18} {code:<4} {(vp or '-'):<14} {(cat or '-'):<12} "
              f"ethukai={ethukai_key(w)} monai={monai_key(w)}")


def cmd_validate(a):
    text = sys.stdin.read() if a.poem in (None, "-") else a.poem
    rep = validate_nerisai_venba(text, strict_ethukai=a.strict)
    print("\n".join(rep.lines()))
    return 0 if rep.valid else 1


def _linker(path):
    from .clinical import ClinicalLinker, load_lexicon
    return ClinicalLinker(load_lexicon(path or data_path("icd10cm_example_lexicon.json")))


def cmd_code(a):
    text = sys.stdin.read() if a.note in (None, "-") else a.note
    L = _linker(a.lexicon)
    findings = L.process(text)
    if a.json:
        print(json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=2))
    else:
        for f in findings:
            print(f"{f.raw_mention:<28} -> {f.entity:<42} {f.match_type:<15} {f.assertion:<15} "
                  f"{f.laterality:<11} {','.join(f.codes) or '-':<18} {f.status}")
        print("allowed codes:", ", ".join(L.allowed_codes(findings)) or "(none)")


def cmd_eval(a):
    from .clinical import evaluate
    r = evaluate(_linker(a.lexicon), a.gold or data_path("gold_assertions.jsonl"))
    for k, v in r.items():
        if k != "errors":
            print(f"{k:<20} {v}")
    for e in r["errors"]:
        print("ERROR:", e["note"], "| gold:", e["gold"], "| pred:", e["pred"])


def cmd_serve(a):
    from .server import serve
    srv = serve(a.host, a.port, a.lexicon)
    print(f"OpenAI-compatible endpoint on http://{a.host}:{a.port}/v1/chat/completions")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_demo_decode(a):
    from .demo import run_demo
    run_demo(seed=a.seed)


def main(argv=None):
    p = argparse.ArgumentParser(prog="yappu", description="Tamil prosody FSM + clinical ICD-10 linker")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("asai", help="show asai split / seer / ethukai / monai for words")
    s.add_argument("words", nargs="+")
    s.set_defaults(fn=cmd_asai)
    s = sub.add_parser("validate", help="validate a Nerisai Venba (text arg or stdin)")
    s.add_argument("poem", nargs="?")
    s.add_argument("--strict", action="store_true", help="exact 2nd-letter Ethukai")
    s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("code", help="extract entities and ICD-10 codes from a note")
    s.add_argument("note", nargs="?")
    s.add_argument("--lexicon")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_code)
    s = sub.add_parser("eval", help="MCC evaluation on a gold JSONL file")
    s.add_argument("--gold")
    s.add_argument("--lexicon")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("serve", help="run the OpenAI-compatible /v1/chat/completions server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--lexicon")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("demo-decode", help="offline constrained-decoding demo (needs torch + transformers)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo_decode)
    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
