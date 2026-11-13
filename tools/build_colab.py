"""Build notebooks/yappu_fsm_colab_single_cell.ipynb: ONE self-contained code cell that
embeds the whole package (source, data, tests), writes it to disk, runs the tests and
every demo. Re-run this script after changing the package."""
import pathlib

import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
INCLUDE = ["pyproject.toml", "README.md", "LICENSE"] + \
    sorted(str(p.relative_to(ROOT)) for p in (ROOT / "yappu_fsm").rglob("*") if p.suffix in (".py", ".json", ".jsonl", ".ace")) + \
    sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("*.py"))

files = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in INCLUDE}
files_src = "FILES = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in files.items()) + "}\n"

HEADER = r'''# ════════════════════════════════════════════════════════════════════════════
#  yappu-fsm · single-cell Colab runner  (Runtime ▸ Run all, or Shift+Enter)
#  Tamil prosody FSM · constrained decoding for HF models · ICD-10-CM (US) linker
#  Everything is embedded below: no upload, no git clone, no model download.
# ════════════════════════════════════════════════════════════════════════════
RUN_TESTS = True          # full pytest suite (~1 min on CPU)
RUN_SERVER_DEMO = True    # OpenAI-compatible /v1/chat/completions on localhost:8000
CODING_SETTING = "outpatient"   # or "inpatient"  (ICD-10-CM Official Guidelines IV.H vs II.H)
RUN_VISION = True         # monochrome YOLOX-Nano + BoT-SORT-style tracking + rule alerts on a sample video
                          # (live laptop camera: run `yappu vision --source 0` on the laptop itself)
'''

BODY = r'''
import importlib, json, os, pathlib, subprocess, sys, threading, urllib.request

BASE = pathlib.Path("/content/yappu-fsm" if os.path.isdir("/content") else "yappu-fsm").resolve()
for rel, text in FILES.items():
    p = BASE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
print(f"wrote {len(FILES)} files to {BASE}")

need = [m for m in ("torch", "transformers", "tokenizers", "pytest", "scipy", "cv2", "onnxruntime")
        if importlib.util.find_spec(m) is None]
need = ["opencv-python" if m == "cv2" else m for m in need]
if need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
for m in [m for m in sys.modules if m == "yappu_fsm" or m.startswith("yappu_fsm.")]:
    del sys.modules[m]   # pick up fresh code when the cell is re-run
if RUN_VISION:           # model + sample video first, so the vision tests can use the real model
    MODEL = pathlib.Path.home() / ".cache" / "yappu" / "yolox_nano.onnx"
    VIDEO = BASE / "vtest.avi"
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    for url, dst in [("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx", MODEL),
                     ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi", VIDEO)]:
        if not dst.exists():
            urllib.request.urlretrieve(url, dst)

def banner(t): print("\n" + "═" * 78 + f"\n  {t}\n" + "═" * 78)

# 1 ── tests ───────────────────────────────────────────────────────────────────
if RUN_TESTS:
    banner("1 · pytest")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=BASE,
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(BASE)})
    print(r.stdout[-1500:] or r.stderr[-1500:])

# 2 ── prosody ─────────────────────────────────────────────────────────────────
from yappu_fsm import validate_nerisai_venba, split_display, encode_asai
banner("2 · Asai splitter and Nerisai Venba validator")
for w in ["கூடினால்", "இலக்கணம்", "கொக்கு", "தவழ்ந்திட"]:
    print(f"  {w:<12} {split_display(w):<14} {encode_asai(w)}")
POEM = """அடக்க முடையா ரறிவிலரென் றெண்ணிக்
கடக்கக் கருதவும் வேண்டா - மடைத்தலையில்
ஓடுமீ னோட உறுமீன் வருமளவும்
வாடி யிருக்குமாம் கொக்கு."""
print("\n".join(validate_nerisai_venba(POEM).lines()))

# 3 ── clinical linker + MCC ──────────────────────────────────────────────────
from yappu_fsm import data_path
from yappu_fsm.clinical import ClinicalLinker, load_lexicon, evaluate
banner(f"3 · ICD-10-CM linker ({CODING_SETTING})")
LEX = load_lexicon(data_path("icd10cm_example_lexicon.json"))
linker = ClinicalLinker(LEX, setting=CODING_SETTING)
NOTE = ("Pt has lt knee osteoarthritus and mild NPDR OD. Denies SOB, but has uncontrolled HTN. "
        "Probable CAP. Father had colon cancer. History of CVA.")
print("note:", NOTE)
for f in linker.process(NOTE):
    print(f"  {f.raw_mention:<28} → {f.entity:<42} {f.assertion:<15} {f.laterality:<11} "
          f"{','.join(f.codes) or '-':<10} {f.status}")
ev = evaluate(ClinicalLinker(LEX), data_path("gold_assertions.jsonl"))
print({k: v for k, v in ev.items() if k != "errors"})

# 3b ── controlled-English coding rules -> Datalog ─────────────────────────────
banner("3b · ICD-10-CM rules in controlled English, compiled to Datalog")
from yappu_fsm.rules import RuleCoder
coder = RuleCoder(ClinicalLinker(LEX), data_path("icd10cm_rules.ace"))
for r in coder.rules[:2]:
    print(f"{r.rid}: {r.paraphrase()}")
res = coder.code("Pt with T2DM and CKD. Also HTN. Probable CAP. Cough for three days.", CODING_SETTING)
print(res.report())

# 4 ── constrained decoding through model.generate() ──────────────────────────
banner("4 · Constrained decoding (random tiny GPT-2: validity comes from the processor)")
from yappu_fsm.demo import run_demo
run_demo(seed=0)

# 5 ── OpenAI-compatible endpoint ─────────────────────────────────────────────
if RUN_SERVER_DEMO:
    banner("5 · OpenAI-compatible /v1/chat/completions")
    from yappu_fsm.server import serve
    try:
        srv = serve(port=8000)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    except OSError:
        print("port 8000 already serving (cell re-run) – reusing it")
    body = {"model": "yappu-icd-linker", "setting": CODING_SETTING,
            "messages": [{"role": "user", "content": "Probable pneumonia. Rt CTS. Family history of HTN."}]}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    print(json.dumps(json.loads(resp["choices"][0]["message"]["content"])["allowed_codes"]))
# 6 ── vision ─────────────────────────────────────────────────────────────────
if RUN_VISION:
    banner("6 · Monochrome YOLOX-Nano → BoT-SORT-style tracker → rule alerts with proofs")
    from yappu_fsm.vision.pipeline import run
    events = run(str(VIDEO), str(MODEL), ["restricted=0,0,0.4,1"], show=False, max_frames=200,
                 save=str(BASE / "vision_out.mp4"), quiet=True)
    for e in events:
        print(f"[{e.t:6.1f}s] {e.kind:<7} {e.event:<15} {'track #' + e.subject if e.subject != 'scene' else 'scene'}")
    raised = next((e for e in events if e.kind == "raised"), None)
    if raised:
        print("proof of the first alert:\n  " + "\n  ".join(raised.proof))
    try:   # show a few annotated monochrome frames inline
        import cv2
        from IPython.display import Image as _Img, display as _display
        cap = cv2.VideoCapture(str(BASE / "vision_out.mp4"))
        for k in (40, 100, 160):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, fr = cap.read()
            if ok:
                _display(_Img(data=cv2.imencode(".jpg", cv2.resize(fr, (576, 432)))[1].tobytes()))
    except Exception as exc:
        print("inline preview skipped:", exc)
    print("annotated video:", BASE / "vision_out.mp4")

print("\ndone ✓")
'''

nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(HEADER + "\n" + files_src + BODY)]
nb.metadata = {"colab": {"provenance": [], "name": "yappu_fsm_colab_single_cell.ipynb"},
               "kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "yappu_fsm_colab_single_cell.ipynb"
out.parent.mkdir(exist_ok=True)
nbformat.write(nb, out)
nbformat.validate(nbformat.read(out, as_version=4))
print(f"{out} · {len(files)} files embedded · {out.stat().st_size // 1024} KB")
