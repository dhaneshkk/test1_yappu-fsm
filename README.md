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
| `yappu_fsm.server` | OpenAI-compatible `/v1/chat/completions` endpoint (stdlib only) |
| `yappu_fsm.cli` | `yappu asai | validate | code | eval | serve | demo-decode` |

## Install

```bash
pip install -e ".[dev]"          # core: pure Python, no dependencies
pip install -e ".[hf,dev]"       # + torch / transformers for constrained decoding
pytest -q                        # 95 tests; HF tests auto-skip without torch
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

* **Assertion:** each entity takes the *nearest preceding* trigger within 6 tokens in its
  clause, or a post-trigger right after it (`ruled out`, `resolved`, `unlikely`).
  Pseudo-triggers such as "no change" and "gram negative" are ignored. Clause walls (`.`, `;`,
  `but`, `however`, …) stop scope from leaking; decimals like `7.5` do not split a clause.
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

Models: `yappu-icd-linker` and `yappu-venba-validator`. Pass `"setting": "inpatient"` in the
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
  assertion model for real charts.
* No model is trained here, so Sharpness-Aware Minimization does not apply. A Lean 4
  formalisation of the decoder guarantee is not included yet; the guarantee is currently
  checked by property tests.

MIT licensed.
