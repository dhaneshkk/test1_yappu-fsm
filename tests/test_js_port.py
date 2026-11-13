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
