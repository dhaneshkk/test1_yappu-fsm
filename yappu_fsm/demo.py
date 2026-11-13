"""Offline constrained-decoding demo.

The Hugging Face hub is not needed: a byte-level BPE tokenizer is trained on the
lexicon and attached to a *randomly initialised* tiny GPT-2.  The model knows
nothing, yet every generation is a valid Nerisai Venba / uses only allowed
ICD-10 codes – the guarantee comes from the logits processors, not the model.
Swap in any real causal LM (``AutoModelForCausalLM.from_pretrained(...)``) and
its tokenizer to get fluent output under the same guarantees.
"""

from __future__ import annotations

from typing import Iterable, Tuple


def build_tiny_model(corpus: Iterable[str], seed: int = 0, vocab_size: int = 400) -> Tuple[object, object]:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<eos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(list(corpus), trainer)
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", pad_token="<eos>")
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=len(hf_tok), n_positions=512, n_embd=32, n_layer=1, n_head=2,
                     eos_token_id=hf_tok.eos_token_id, bos_token_id=hf_tok.eos_token_id,
                     pad_token_id=hf_tok.eos_token_id)
    model = GPT2LMHeadModel(cfg).eval()
    return hf_tok, model


def generate_venba(tokenizer, model, lexicon=None, prompt: str = "வெண்பா:", do_sample: bool = True,
                   seed: int = 0, max_new_tokens: int = 400):
    import torch
    from transformers import LogitsProcessorList

    from .decoding import VenbaDecoder, VenbaLogitsProcessor
    from .lexicon import VenbaLexicon

    lexicon = lexicon or VenbaLexicon.sample()
    proc = VenbaLogitsProcessor(tokenizer, VenbaDecoder(lexicon))
    ids = tokenizer(prompt, return_tensors="pt")
    torch.manual_seed(seed)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=0,
                         logits_processor=LogitsProcessorList([proc]),
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    gen = out[0, ids["input_ids"].shape[1]:].tolist()
    state = proc.state_for(gen)
    words = proc.decoder.finished_words(state) if state is not None else []
    return words, tokenizer.decode(gen, skip_special_tokens=True)


def generate_codes(tokenizer, model, allowed_codes, prompt: str, seed: int = 0, max_new_tokens: int = 80):
    import torch
    from transformers import LogitsProcessorList

    from .decoding import ICD10LogitsProcessor, parse_codes

    proc = ICD10LogitsProcessor(tokenizer, allowed_codes)
    ids = tokenizer(prompt, return_tensors="pt")
    torch.manual_seed(seed)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True, top_k=0,
                         logits_processor=LogitsProcessorList([proc]),
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return parse_codes(text), text


def run_demo(seed: int = 0):
    from . import data_path
    from .clinical import ClinicalLinker, load_lexicon
    from .lexicon import SAMPLE_WORDS
    from .prosody import format_venba, validate_nerisai_venba

    lex = load_lexicon(data_path("icd10cm_example_lexicon.json"))
    codes_corpus = [c for v in lex.values() for vv in v.values() for c in ([vv] if isinstance(vv, str) else vv)]
    corpus = [" ".join(SAMPLE_WORDS)] * 4 + ["CODES: " + "; ".join(codes_corpus)] * 4
    tok, model = build_tiny_model(corpus, seed=seed)

    print("== Constrained Nerisai Venba from a random tiny GPT-2 ==")
    words, _ = generate_venba(tok, model, seed=seed)
    print(format_venba(words))
    rep = validate_nerisai_venba(words)
    print("validator:", "VALID" if rep.valid else "INVALID", rep.vikarpam or "")

    print("\n== Constrained ICD-10 emission ==")
    linker = ClinicalLinker(lex)
    note = "Pt has lt knee osteoarthritis. Denies SOB, but has uncontrolled HTN. R/O CAP."
    allowed = linker.allowed_codes(linker.process(note))
    codes, text = generate_codes(tok, model, allowed, prompt=f"Note: {note}\nCODES:", seed=seed)
    print("note:          ", note)
    print("allowed codes: ", allowed)
    print("model emitted: ", codes)
