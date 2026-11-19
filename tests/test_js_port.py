"""The tutorial's JavaScript port must agree with the Python implementation."""
import json
import pathlib
import shutil
import subprocess

import pytest
from conftest import NANNERI, POEM_1, POEM_2

from yappu_fsm import encode_asai, ethukai_key, monai_key, split_display, validate_nerisai_venba
from yappu_fsm.lexicon import SAMPLE_WORDS
from yappu_fsm.prosody import eetrucheer_vaipadu

JS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "src" / "prosody.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None or not JS.exists(), reason="node or docs/src missing")

WORDS = sorted(set(SAMPLE_WORDS) | {"கூவிளங்கனி", "கொக்கு", "ஃபேன்", "தமிழ்", "சங்கம்", "ஔவை"})
BROKEN = POEM_2.replace("நிலவொளி", "வானில்")


def test_js_matches_python():
    script = f"""
const Y = require({json.dumps(str(JS))});
const words = {json.dumps(WORDS, ensure_ascii=False)};
const poems = {json.dumps([POEM_1, POEM_2, NANNERI, BROKEN], ensure_ascii=False)};
console.log(JSON.stringify({{
  w: words.map(w => [Y.encode(w), Y.split(w), Y.ethukai(w), Y.monai(w), Y.eetrucheer(w)]),
  p: poems.map(p => {{ const r = Y.validate(p); return [r.valid, r.vikarpam, r.feet.map(f => f.thalai)]; }})
}}));"""
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    for w, js in zip(WORDS, out["w"]):
        assert js == [encode_asai(w), split_display(w), ethukai_key(w), monai_key(w), eetrucheer_vaipadu(w)], w
    for p, js in zip([POEM_1, POEM_2, NANNERI, BROKEN], out["p"]):
        r = validate_nerisai_venba(p)
        assert js == [r.valid, r.vikarpam, [f.thalai_in for f in r.feet]]


RULES_JS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "src" / "rules.js"


@pytest.mark.skipif(not RULES_JS.exists(), reason="docs/src missing")
def test_js_rules_match_python():
    """The tutorial's in-browser Datalog + ordering must give the same codes as RuleCoder,
    for every demo note, both settings, and with each rule switched off."""
    import importlib.util
    import sys
    tools = pathlib.Path(__file__).resolve().parents[1] / "tools" / "export_tutorial_data.py"
    spec = importlib.util.spec_from_file_location("export_tutorial_data", tools)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_tutorial_data"] = mod
    spec.loader.exec_module(mod)
    d = mod.rules_data()
    script = f"""
const RJ = require({json.dumps(str(RULES_JS))});
const d = {json.dumps(d, ensure_ascii=False)};
const out = {{normal: [], ablation: {{}}}};
for (const n of d.notes) for (const s of ["outpatient", "inpatient"]) {{
  const r = RJ.code(d.clauses, n.facts[s], n.findings);
  out.normal.push({{codes: r.codes, blocked: r.blocked, conflicts: r.conflicts}});
}}
for (const rid of Object.keys(d.ablation)) {{
  out.ablation[rid] = {{}};
  for (const n of d.notes) out.ablation[rid][n.note] = RJ.code(d.clauses, n.facts.inpatient, n.findings, new Set([rid])).codes;
}}
console.log(JSON.stringify(out));"""
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    expected = [n["expected"][s] for n in d["notes"] for s in ("outpatient", "inpatient")]
    assert out["normal"] == expected
    assert out["ablation"] == d["ablation"]


VISION_JS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "src" / "vision.js"


@pytest.mark.skipif(not VISION_JS.exists(), reason="docs/src missing")
def test_js_timer_matches_python():
    """The tutorial's alert simulator must reproduce Timer + AlertMachine frame by frame."""
    pytest.importorskip("numpy")           # the reference comes from yappu_fsm.vision (vision extra)
    import importlib.util
    import sys
    tools = pathlib.Path(__file__).resolve().parents[1] / "tools" / "export_tutorial_data.py"
    spec = importlib.util.spec_from_file_location("export_tutorial_data_v", tools)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_tutorial_data_v"] = mod
    spec.loader.exec_module(mod)
    ref = mod.timer_reference()
    script = f"""
const VJ = require({json.dumps(str(VISION_JS))});
const ref = {json.dumps(ref)};
const out = {{}};
for (const key of Object.keys(ref.expected)) {{
  const [g, thr] = key.split("|").map(Number);
  out[key] = VJ.simulate(ref.seen, ref.fps, g, thr, 0).active;
}}
console.log(JSON.stringify(out));"""
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out == ref["expected"]
