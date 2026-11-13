"""End-to-end tests through transformers' model.generate() with a locally trained
byte-level BPE tokenizer and a randomly initialised tiny GPT-2 (no downloads)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from yappu_fsm import data_path  # noqa: E402
from yappu_fsm.clinical import ClinicalLinker, load_lexicon  # noqa: E402
from yappu_fsm.decoding import ICD10LogitsProcessor, VenbaDecoder, VenbaLogitsProcessor, token_bytes_table  # noqa: E402
from yappu_fsm.demo import build_tiny_model, generate_codes, generate_venba  # noqa: E402
from yappu_fsm.lexicon import SAMPLE_WORDS, VenbaLexicon  # noqa: E402
from yappu_fsm.prosody import validate_nerisai_venba  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    lex = load_lexicon(data_path("icd10cm_example_lexicon.json"))
    codes = [c for v in lex.values() for vv in v.values() for c in ([vv] if isinstance(vv, str) else vv)]
    corpus = [" ".join(SAMPLE_WORDS)] * 4 + ["CODES: " + "; ".join(codes)] * 4
    return build_tiny_model(corpus, seed=0)


def test_token_bytes_roundtrip(tiny):
    tok, _ = tiny
    table = token_bytes_table(tok)
    ids = tok("வண்ண நிலவொளி", add_special_tokens=False)["input_ids"]
    assert b"".join(table[i] for i in ids).decode("utf-8") == "வண்ண நிலவொளி"
    assert table[tok.eos_token_id] is None


@pytest.mark.parametrize("seed", range(4))
def test_generate_valid_venba(tiny, seed):
    tok, model = tiny
    words, _ = generate_venba(tok, model, seed=seed)
    rep = validate_nerisai_venba(words)
    assert len(words) == 15 and rep.valid, rep.lines()


def test_monai_boost_applied(tiny):
    tok, _ = tiny
    proc = VenbaLogitsProcessor(tok, VenbaDecoder(VenbaLexicon.sample(), monai_boost=3.0))
    dec = proc.decoder
    s = dec.advance(dec.initial(), "வண்ண".encode())
    s = dec.advance(s, b" ")
    _, boosted, pozhippu = proc.allowed(s)
    # foot 1 of line 1: tokens starting a word whose first letter is in the ம/வ + அ group
    assert boosted and not pozhippu


def test_processor_masks_logits(tiny):
    tok, _ = tiny
    proc = VenbaLogitsProcessor(tok, VenbaDecoder(VenbaLexicon.sample()))
    ids = tok("வெண்பா:", return_tensors="pt")["input_ids"]
    scores = torch.zeros((1, len(tok)))
    out = proc(ids, scores.clone())
    allowed = torch.isfinite(out[0]).nonzero().flatten().tolist()
    assert allowed and len(allowed) < len(tok)
    for tid in allowed:
        assert proc.decoder.advance(proc.decoder.initial(), proc.token_bytes[tid]) is not None


@pytest.mark.parametrize("note", [
    "Pt has lt knee osteoarthritis. Denies SOB, but has uncontrolled HTN. R/O CAP.",
    "Bilateral hip pain. History of MI.",
    "Mild NPDR OD and diabetic nefropathy.",
])
def test_generated_codes_subset_of_allowed(tiny, lexicon, note):
    tok, model = tiny
    linker = ClinicalLinker(lexicon)
    allowed = linker.allowed_codes(linker.process(note))
    assert allowed
    for seed in range(10):
        codes, _ = generate_codes(tok, model, allowed, prompt=f"Note: {note}\nCODES:", seed=seed)
        assert codes and set(codes) <= set(allowed) and len(codes) == len(set(codes))


def test_icd_processor_ignores_text_before_marker(tiny):
    tok, _ = tiny
    proc = ICD10LogitsProcessor(tok, ["I10"])
    ids = tok("free text", return_tensors="pt")["input_ids"]
    scores = torch.zeros((1, len(tok)))
    assert torch.equal(proc(ids, scores.clone()), scores)
