# yappu-fsm

Tamil prosody (**Yappu Ilakkanam**, யாப்பிலக்கணம்) as finite-state machines, constrained
decoding for Hugging Face LLMs, and a rule-based clinical entity linker for ICD-10 coding.

This repository consolidates and corrects all the code from the "Tamil Prosody → FSM →
constrained decoding → clinical ICD-10 coding" exploration into one tested package.

| Module | What it does |
|---|---|
| `yappu_fsm.tamil` | Graphemes, Kuril/Nedil/Otru classes, Asai encoding (`1` = Ner, `0` = Nirai), Ethukai and Monai keys |
| `yappu_fsm.prosody` | Seer table, Thalai naming, full **Nerisai Venba validator** (15 feet, Vendalai, Eetrucheer incl. Kaasu/Pirappu, Oru/Iru Vikarpam Ethukai) |
| `yappu_fsm.lexicon` | Annotated word lexicon with dead-end-free lookahead |
| `yappu_fsm.decoding` | Byte-trie state machines + `VenbaLogitsProcessor` and `ICD10LogitsProcessor` for `model.generate()` |
| `yappu_fsm.clinical` | Abbreviations, clause walls, NegEx-style assertion, closest-token laterality, phonetic/fuzzy matching, ICD-10 lookup, MCC evaluation |
| `yappu_fsm.rules` | ICD-10-CM coding rules in ACE-style controlled English, compiled to stratified Datalog, with proof traces |
| `yappu_fsm.vision` | Monochrome YOLOX-Nano (onnxruntime), BoT-SORT-style tracker, time-window event rules → Datalog alerts with proofs |
| `yappu_fsm.server` | OpenAI-compatible `/v1/chat/completions` endpoint (stdlib only) |
| `yappu_fsm.cli` | `yappu asai | validate | code [--rules] | rules | eval | serve | vision | demo-decode` |

## Install

```bash
pip install -e ".[dev]"          # core: pure Python, no dependencies
pip install -e ".[hf,dev]"       # + torch / transformers for constrained decoding
pytest -q                        # 159 tests; HF tests auto-skip without torch
```

Colab: `notebooks/yappu_fsm_colab_single_cell.ipynb` is one self-contained cell. It embeds the
whole package, so nothing needs uploading; run it to get the tests and every demo.
`docs/tutorial.html` is a progressive, interactive tutorial with comics.

## Prosody

```bash
$ yappu asai கூடினால் இலக்கணம் கொக்கு
கூடினால்    கூ/டினால்    10   Koovilam    Vilacheer   ethukai=N:ட monai=C:க:U
இலக்கணம்    இலக்/கணம்    00   Karuvilam   Vilacheer   ethukai=K:ல monai=V:I
கொக்கு      கொக்/கு      11   Thema       Maacheer    ethukai=K:க monai=C:க:U   (Kaasu as final foot)

$ yappu validate "அடக்க முடையா ரறிவிலரென் றெண்ணிக் ... வாடி யிருக்குமாம் கொக்கு."
...
Verdict: VALID Nerisai Venba (Iru Vikarpam)
```

```python
from yappu_fsm import validate_nerisai_venba
rep = validate_nerisai_venba(poem_text)      # or a list of 15 words
print("\n".join(rep.lines()))
```

Ethukai uses the traditional rule by default: the 2nd letters share a consonant and the
1st letters agree in length. `strict_ethukai=True` requires the exact same 2nd letter.

## Constrained decoding (Hugging Face)

LLM tokens are sub-word pieces, often partial UTF-8 bytes for Tamil, so the Asai of a single
token is meaningless. Both processors map every vocabulary token to its raw bytes
(byte-level BPE and SentencePiece are both handled) and walk a byte trie of allowed whole
words or codes. A token is unmasked only if the text stays on a completable path. Allowed
sets are cached per decoder state.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from yappu_fsm import VenbaLexicon, format_venba
from yappu_fsm.decoding import VenbaDecoder, VenbaLogitsProcessor

tok = AutoTokenizer.from_pretrained("your/causal-lm")
model = AutoModelForCausalLM.from_pretrained("your/causal-lm")
lexicon = VenbaLexicon(my_tamil_word_list)          # bigger lexicon -> better poems
proc = VenbaLogitsProcessor(tok, VenbaDecoder(lexicon, monai_boost=3.0))

ids = tok("வெண்பா:", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=400, do_sample=True,
                     logits_processor=LogitsProcessorList([proc]), eos_token_id=tok.eos_token_id)
state = proc.state_for(out[0, ids["input_ids"].shape[1]:].tolist())
print(format_venba(proc.decoder.finished_words(state)))
```

Hard constraints are applied as `-inf` masks: Vendalai, no Kanicheer or 4-asai feet, Ethukai
at feet 4, 7 and 12, and a valid Eetrucheer. Monai is a soft boost, ×1.5 at the 3rd foot
(Pozhippu Monai). The lexicon precomputes which words keep a full poem reachable, so
generation never dead-ends.

ICD-10: put a marker at the end of the prompt and only allowed codes can follow:

```python
from yappu_fsm.decoding import ICD10LogitsProcessor, parse_codes
allowed = linker.allowed_codes(linker.process(note))   # codes from the rule-based linker
proc = ICD10LogitsProcessor(tok, allowed)               # "CODES: X; Y" then newline / EOS
```

`yappu demo-decode` runs both processors offline. It trains a byte-level BPE tokenizer on the
lexicon and uses a **randomly initialised** tiny GPT-2, which shows that validity comes from
the processor, not the model.

**What is guaranteed.** Every completed generation is in the allowed language. For Venba,
`validate_nerisai_venba` returns VALID. For ICD-10, every code is in the allowed set and none
repeats. The test suite checks this by property tests over random decoder paths and through
`model.generate()`. The processors do **not** guarantee that a code is clinically correct,
only that it came from the allowed set.

## Clinical linker

```bash
$ yappu code "Pt has lt knee osteoarthritus and mild NPDR OD. Denies SOB, but has uncontrolled HTN. R/O CAP."
knee osteoarthritus  -> knee osteoarthritis                        PHONETIC_FUZZY PRESENT   LEFT        M17.12   CODE
mild nonproliferative diabetic retinopathy -> (same)               EXACT          PRESENT   RIGHT       E11.3291 CODE
shortness of breath  -> shortness of breath                        EXACT          NEGATED   UNSPECIFIED -        DO_NOT_CODE
hypertension         -> hypertension                               EXACT          PRESENT   UNSPECIFIED I10      CODE
pneumonia            -> pneumonia                                  EXACT          UNCERTAIN UNSPECIFIED -        DO_NOT_CODE
allowed codes: M17.12, E11.3291, I10
```

* **Assertion (NegEx + ConText):**
  * Pre-triggers ("denies", "no", "without", "ruled out", "absence of", "no history of", …)
    scan forward, and post-triggers ("was ruled out", "not demonstrated", "never developed",
    "absent", "cannot be ruled out", "unlikely", "resolved") scan backward. Both reach at most 6
    tokens and stop at CONJ terminators ("aside from", "secondary to", "due to", "except", …).
  * Pseudo-triggers ("no change", "gram negative", "not ruled out") cannot open a negation.
  * ConText adds HISTORICAL, FAMILY_HISTORY and HYPOTHETICAL ("return if", "if the patient
    develops", "watch for").
  * Clause walls (`.`, `;`, `but`, `however`, …) split the note first; decimals like `7.5` do not.
* **Laterality:** closest-token binding within 4 tokens, with each modifier used once.
  Bilateral hip pain maps to both side codes, because no single bilateral code exists.
* **Fuzzy matching:** compared word by word. A word matches exactly, or, if it is at least 5
  letters, by phonetic skeleton or edit similarity ≥ 0.84. So "osteoarthritus" and "pnemonia"
  match, but "hip rain" does not match "hip pain".
* **Abbreviations:** ambiguous ones (CAP, OD, OS, OU, MI) are case-sensitive, so "knee cap"
  is not pneumonia.

### Evaluation (MCC)

```bash
$ yappu eval
n_pairs              31
assertion_mcc        0.9558
laterality_mcc       1.0
codable_mcc          0.9269
```

NegEx alignment, measured on `data/gold_negex_heldout.jsonl` (12 notes written after the
rules were tuned on `gold_negex.jsonl`, scored once before any further change):

| | assertion MCC | codable MCC |
|---|---|---|
| previous linker | 0.619 | 0.539 |
| NegEx/ConText linker | 0.861 | 0.811 |

The two misses that run exposed ("cannot be ruled out" after the term; "which" wrongly used
as a terminator) were then fixed, so that set is now a regression test, not a held-out one.
The next honest measurement needs real notes.

`data/gold_assertions.jsonl` is a **small synthetic smoke-test set** of 20 notes written for
regression testing, not a clinical validation. Missed and spurious entities count as their
own classes, so recall errors lower the MCC. The remaining error ("chest pain is unlikely
to be cardiac" → UNCERTAIN) is kept as a known limitation of rule-based scope.

### ICD-10-CM (US) coding rules

The linker targets **ICD-10-CM** and follows the *ICD-10-CM Official Guidelines for Coding
and Reporting* for uncertain diagnoses:

| Setting | Guideline | "Probable / suspected / rule out X" |
|---|---|---|
| `outpatient` (default) | Section IV.H | not coded (`DO_NOT_CODE`); code documented signs and symptoms instead |
| `inpatient` | Section II.H / III.C | coded as if established (assertion stays `UNCERTAIN` for audit) |

Negated findings are never coded. History maps to Z86/Z87 personal-history codes and family
history to Z80–Z84 where the lexicon provides them.

```bash
yappu code "Probable pneumonia, start antibiotics."                      # DO_NOT_CODE
yappu code "Probable pneumonia, start antibiotics." --setting inpatient  # J18.9
```

`data/icd10cm_example_lexicon.json` holds 32 common conditions (diabetes, hypertension, AF,
heart failure, CKD, COPD, asthma, pneumonia, OA, joint pain with laterality, carpal tunnel,
history and family-history codes, and more). It is a demo subset. Extend it in the same
format (`{"entity": {"ASSERTION_LATERALITY": code | [codes]}}`) and verify codes against the
current CMS release. All output is decision support for a certified coder.

## Coding rules in controlled English → Datalog

Conventions that relate facts to each other (combination codes, "code first / use additional",
Excludes1, symptoms integral to a diagnosis, setting-dependent guidelines) are written as
ACE-style sentences in `data/icd10cm_rules.ace`:

```text
DM_CKD:  If the patient has a type-2-diabetes and the patient has a chronic-kidney-disease
         then the code E11.22 is assigned and the code E11.22 precedes the code N18.9.
HTN_CKD: If the patient has a hypertension and the patient has a chronic-kidney-disease
         then the code I12.9 is assigned and the code I10 is not assigned and the code I12.9 precedes the code N18.9.
EXCL_E10_E11: If the code E10.9 is assigned and the code E11.9 is assigned then the codes E10.9 and E11.9 conflict.
INPT_UNCERTAIN: If the encounter is an inpatient-encounter and the patient has an uncertain diagnosis X
         then the candidate code of X is assigned.
```

```bash
yappu rules             # parse, show the deterministic paraphrase of every rule
yappu rules --datalog   # the compiled program, ready to load into another Datalog engine
yappu code "Pt with T2DM and CKD. Also HTN." --rules     # E11.22, I12.9, N18.9 with proofs
```

How it works:

1. The linker turns the note into facts: `has(patient, type-2-diabetes)`, `uncertain(patient, pneumonia)`,
   `side(...)`, `candidate(patient, concept, code)`, `code(patient, code)` and `setting(patient, inpatient)`.
2. Rules are parsed **once** by `yappu_fsm.rules.ace`. Content words must be lexicon concepts,
   anything outside the grammar is rejected with the exact word position, and each rule gets a
   bracketed paraphrase for review. Each rule compiles to Horn clauses.
3. `yappu_fsm.rules.datalog` evaluates the program: semi-naive, with stratified negation
   (`final(P, C) :- code(P, C), not blocked(P, C).`). Unsafe or unstratifiable rule sets are
   rejected at load time, so evaluation always terminates.
4. The result holds the final codes (sequenced by `precedes`), blocked codes, Excludes1
   conflicts for coder review, and a proof tree per code that cites the rule sentence.

**Sequencing.** Each final code starts at a default position. A code the linker produced keeps
its finding's place. A rule-derived code takes the place of the code it replaced (E11.22 where
E11.9 was, I12.9 where I10 was); otherwise it sits at the earliest finding its proof mentions.
`precedes` facts are then applied as a stable topological order. House sequencing policy is
itself written as rules, using code variables and code kinds (diagnosis, symptom R, factor Z,
external cause V–Y):

```text
SEQ_DX_SYMPTOM: If a code C is assigned and the code C is a symptom code and a code D is assigned
                and the code D is a diagnosis code then the code D precedes the code C.
```

`"Pt with T2DM and CKD. Also HTN. Probable CAP. Cough for three days."` → `E11.22, I12.9, N18.9, R05.9`.
This is a default order for review; principal-diagnosis selection (Section II) is not modelled.

The language is a coding-rule **subset in the spirit of ACE** and is parsed by this package; it
does not call the Attempto Parsing Engine. It has no wildcards over code families (E11.9 needs
one blocking rule per complication code).

### Performance

Measured on CPU, Python 3.13 (`tests/test_clinical.py::test_index_scales` guards this):

| | ms / note | notes / s |
|---|---|---|
| linker, 32-entry demo lexicon | 0.14 | ~7,400 |
| linker + 16 coding rules (Datalog) | 0.30 | ~3,300 |
| linker, 100,000-entry synthetic lexicon | 0.42 | ~2,400 |

The linker finds candidate entries through hash indexes on each entry's first word: exact
form, phonetic skeleton, and one-deletion keys, which cover one substitution, insertion,
deletion or swap. A full scan is still available (`ClinicalLinker(index=False)`), and a test
checks that both give identical results on the gold notes and 300 random typos. A first word
more than two edits away from the lexicon term is not retrieved by the index.

## Vision: monochrome laptop camera → YOLOX-Nano → tracking → rule alerts

```bash
pip install -e ".[vision]"            # onnxruntime, opencv-python, numpy, scipy (no PyTorch)
yappu vision --download               # yolox_nano.onnx (3.7 MB) -> ~/.cache/yappu/
yappu vision --source 0 --zone restricted=0,0,0.4,1        # laptop camera, press q to quit
yappu vision --source clip.mp4 --no-show --save out.mp4 --log alerts.jsonl
```

The pipeline, per frame:

1. **Monochrome.** The frame is converted to grey, with optional CLAHE (`--clahe`) for dim
   rooms. The single channel is copied to YOLOX's three input channels, and the preview stays grey.
2. **YOLOX-Nano** (official ONNX export, 416×416) runs under onnxruntime: grid decode,
   objectness × class score, class-wise NMS. The score threshold is kept low (0.1) so the
   tracker's second stage gets the weak boxes.
3. **BoT-SORT-style tracker** (`vision/tracker.py`, an MIT-licensed reimplementation):
   * a Kalman filter on (x, y, w, h) with size-proportional noise;
   * ByteTrack two-stage association: high-score boxes by IoU × score, then low-score boxes by plain IoU;
   * optional camera-motion compensation (`--gmc`);
   * a lifecycle state machine TENTATIVE → CONFIRMED ⇄ LOST → REMOVED, logged per track.

   No ReID network is used, since a static laptop camera rarely needs it.
4. **Temporal facts.** Each track produces `in_zone`, `holds` (an object box at least 50 % inside the
   person box) and `visible` facts. A timer turns them into duration facts
   (`in_zone_for(7, restricted, 2.0)`). Gaps shorter than 0.7 s, such as a missed
   detection or a brief occlusion, don't reset the timer. This is the video counterpart of a NegEx scope window.
5. **Rules as sentences** (`data/vision_rules.ace`) compile to the same Datalog engine:

   ```text
   INTRUSION: If a person T is in the restricted-zone for at least 2 seconds then a zone-intrusion alert is raised for the person T.
   PHONE:     If the person T holds a cell-phone for at least 3 seconds then a phone-use alert is raised for the person T.
   CROWD:     If at least 2 persons are in the restricted-zone then a crowding alert is raised.
   ```

   Unknown COCO classes and unknown zones are rejected when the rules load.
6. **Alert state machine** IDLE → ACTIVE → COOLING → IDLE: each alert is raised once with its
   proof and cleared after a 3 s cool-down.

Measured on the sandbox CPU with OpenCV's `vtest.avi` pedestrian clip converted to grey: detection
≈ 21 ms/frame, tracking 0.5–2 ms, rules ≈ 0.1 ms. On a synthetic stay/pass-through
scenario where the detector misses 25 % of frames, the time-window rules reach MCC **0.941**
against **0.685** for per-frame alerting (`tests/test_vision.py`).

Rules should fire on **positive evidence**. "Does not hold" is allowed in the rule language, but in
video a missing detection may only be occlusion or a dark frame. COCO has no helmet or vest
classes, so PPE rules need a fine-tuned YOLOX class (for example `head_no_helmet`) trained on
grey frames, using Sharpness-Aware Minimization per the project standard. COCO weights see
colour during training, so expect lower recall on grey input until then. Tracking quality
(HOTA, IDF1, ID switches) isn't measured here because there's no ground truth; MCC applies to the
alert layer.

## OpenAI-compatible endpoint

```bash
yappu serve --port 8000
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
r = c.chat.completions.create(model="yappu-icd-linker",
        messages=[{"role": "user", "content": "Mild NPDR OD, family history of HTN."}])
print(r.choices[0].message.content)    # JSON: findings + allowed_codes
```

Models: `yappu-icd-linker`, `yappu-icd-rules` (codes + proofs) and `yappu-venba-validator`. Pass `"setting": "inpatient"` in the
request body (`extra_body` in the OpenAI SDK) for the inpatient rule. Streaming is not supported.

## Corrections relative to the original thread

* Asai: kuril + kuril is one Nirai (கூ/டினால், இலக்/கணம், தவழ்ந்/திட, ஏற்/றிடும், பார்ப்/பவர்);
  NFC normalisation; aytham handled.
* Kaasu/Pirappu final feet are accepted (kutriyalukaram), so the classical Nanneri verse validates.
* Ethukai compares the consonant of the 2nd letter plus 1st-letter length; the old exact-letter
  rule rejected valid poems.
* Monai consonant pairs follow Ina Ezhuthu: ச/த, ஞ/ந, ம/வ (not த/ந, ச/ய).
* `SEER_CATEGORIES` filtered tuples against a string, so Kanicheer was never excluded.
* Crashes fixed: `range(input_ids.shape)`, `extract_ethukai_key(words)` with a list,
  `letters` returned instead of `letters[1]`, and a missing operator in the laterality binder.
* Logits processors work on whole words and codes via byte tries instead of single tokens.
  The ICD processor no longer loops over the full vocabulary in Python on every step
  (results are cached), and duplicate codes can no longer be emitted.
* Clinical: nearest-trigger assertion (the old version marked "Denies SOB, history of MI" as
  HISTORICAL for SOB), trigger scope limit, pseudo-triggers, "resolved" → HISTORICAL.
  Corrected CM mappings: mild NPDR is E11.329x without macular edema; E11.319 has no
  laterality; there is no bilateral M25.55x code.
* Tests that indexed `results` instead of `results[0]`, and the "hip rain" guardrail test
  that could not pass, are fixed.

## Limitations

* The English Asai encoder is a rough heuristic. Real English meter needs a pronouncing
  dictionary.
* Venba quality depends on the lexicon. The bundled one (~55 words) is for tests.
* The clinical linker is rule-based and lexicon-driven. Pair it with a trained NER and
  assertion model for real charts. Trigger scope is clause-based: in "Probable CAP with cough
  and fever" the symptoms are also read as uncertain (see `test_known_limitation_*`).
* No model is trained here, so Sharpness-Aware Minimization does not apply. A Lean 4
  formalisation of the decoder guarantee is not included yet; the guarantee is currently
  checked by property tests.

MIT licensed.
