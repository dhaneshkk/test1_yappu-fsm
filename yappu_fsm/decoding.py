"""Constrained decoding for Hugging Face ``model.generate()``.

Two logits processors are provided:

* :class:`VenbaLogitsProcessor` – forces the generated continuation to be a
  complete Nerisai Venba built from a :class:`~yappu_fsm.lexicon.VenbaLexicon`
  (Vendalai linking, no Kanicheer, Ethukai rhyme, valid Eetrucheer) and adds a
  soft logit boost for Monai alliteration.
* :class:`ICD10LogitsProcessor` – after a marker such as ``"CODES:"`` the model
  can only emit codes from an allowed set (e.g. the codes produced by the
  clinical linker), separated by ``";"`` and ended by a newline or EOS.

Why word-level?  LLM tokens are subword pieces (often partial UTF-8 bytes for
Tamil), so the Asai of a single token says nothing about the word being built.
Both processors therefore work on the **bytes** of every vocabulary token and
walk a byte trie of whole allowed words / codes.  A token is allowed only if,
appended to the text generated so far, it stays on a path that can still be
completed.  Allowed-token sets are cached per decoder state.

Formal property (checked by the tests): every sequence accepted by the decoder
is a member of the allowed language – for Venba, ``validate_nerisai_venba``
returns VALID; for ICD-10, every emitted code is in the allowed set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .lexicon import Entry, VenbaLexicon
from .prosody import LINE_STARTS
from .tamil import monai_key

SPACE_BYTES = {0x20, 0x0A}


# ==============================================================================
# Token -> raw bytes
# ==============================================================================


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 byte <-> printable unicode table (used by byte-level BPE)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _decoder_types(tokenizer) -> set:
    try:
        spec = json.loads(tokenizer.backend_tokenizer.to_str())
    except Exception:
        return set()
    types = set()

    def walk(node):
        if isinstance(node, dict):
            if "type" in node:
                types.add(node["type"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec.get("decoder"))
    walk(spec.get("pre_tokenizer"))
    return types


def token_bytes_table(tokenizer) -> List[Optional[bytes]]:
    """Raw bytes produced by each token id (``None`` for special tokens)."""
    vocab_size = len(tokenizer)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    types = _decoder_types(tokenizer)
    table: List[Optional[bytes]] = [None] * vocab_size
    if "ByteLevel" in types:
        inv = {c: b for b, c in _bytes_to_unicode().items()}
        for tid in range(vocab_size):
            if tid in special:
                continue
            tok = tokenizer.convert_ids_to_tokens(tid)
            if tok is None:
                continue
            try:
                table[tid] = bytes(inv[c] for c in tok)
            except KeyError:
                table[tid] = tok.encode("utf-8")
        return table
    for tid in range(vocab_size):
        if tid in special:
            continue
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tok is None:
            continue
        if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
            table[tid] = bytes([int(tok[3:5], 16)])  # SentencePiece byte fallback
        elif "Metaspace" in types or "▁" in tok:
            table[tid] = tok.replace("▁", " ").encode("utf-8")
        else:
            table[tid] = tokenizer.decode([tid]).encode("utf-8")
    return table


# ==============================================================================
# Byte trie
# ==============================================================================


class TrieNode:
    __slots__ = ("children", "value", "monai", "nid")

    def __init__(self, nid: int):
        self.children: Dict[int, "TrieNode"] = {}
        self.value = None          # Entry (Venba) or str (ICD code) at a terminal node
        self.monai: set = set()    # Monai keys of words in this subtree
        self.nid = nid


class ByteTrie:
    def __init__(self, items: Iterable[Tuple[str, object]]):
        self._n = 0
        self.root = self._new()
        for text, value in items:
            node = self.root
            mk = monai_key(text)
            node.monai.add(mk)
            for b in text.encode("utf-8"):
                node = node.children.get(b) or node.children.setdefault(b, self._new())
                node.monai.add(mk)
            node.value = value

    def _new(self) -> TrieNode:
        self._n += 1
        return TrieNode(self._n)


# ==============================================================================
# Venba decoder state machine
# ==============================================================================


@dataclass(frozen=True)
class VenbaState:
    k: int = 0                         # index of the foot being written (0..14)
    expected: Optional[str] = None     # required first asai of this foot
    key_a: Optional[str] = None        # Ethukai anchor of line 1
    key_b: Optional[str] = None        # Ethukai anchor of line 3
    line_monai: Optional[str] = None   # Monai key of the current line's first word
    node_path: bytes = b""             # bytes of the current (partial) word
    lead_space_ok: bool = True         # one leading space allowed before foot 0
    words: Tuple[str, ...] = field(default_factory=tuple, compare=False)


class VenbaDecoder:
    """Pure-Python state machine; usable without torch."""

    def __init__(self, lexicon: VenbaLexicon, monai_boost: float = 3.0, pozhippu_factor: float = 1.5):
        self.lex = lexicon
        self.monai_boost = monai_boost
        self.pozhippu_factor = pozhippu_factor
        self._tries: Dict[tuple, ByteTrie] = {}

    def trie_for(self, s: VenbaState) -> ByteTrie:
        sig = (s.k, s.expected, s.key_a, s.key_b)
        t = self._tries.get(sig)
        if t is None:
            t = ByteTrie((e.word, e) for e in self.lex.candidates(*sig))
            self._tries[sig] = t
        return t

    def node(self, s: VenbaState) -> Optional[TrieNode]:
        n = self.trie_for(s).root
        for b in s.node_path:
            n = n.children.get(b)
            if n is None:
                return None
        return n

    def initial(self) -> VenbaState:
        return VenbaState()

    def advance(self, s: VenbaState, data: bytes) -> Optional[VenbaState]:
        node = self.node(s)
        if node is None:
            return None
        for b in data:
            if b in SPACE_BYTES:
                if not s.node_path:
                    if s.k == 0 and s.lead_space_ok and not s.words:
                        s = replace(s, lead_space_ok=False)
                        continue
                    return None
                e: Entry = node.value
                if e is None or s.k == 14:
                    return None
                k = s.k
                key_a = e.ethukai if k == 0 else s.key_a
                key_b = e.ethukai if k == 8 else s.key_b
                nk = k + 1
                line_monai = e.monai if k in LINE_STARTS else s.line_monai
                s = VenbaState(nk, self.lex._next_keys(k, e, s.key_a, s.key_b)[0], key_a, key_b,
                               line_monai, b"", False, s.words + (e.word,))
                node = self.trie_for(s).root
                if not node.children:
                    return None
            else:
                node = node.children.get(b)
                if node is None:
                    return None
                s = replace(s, node_path=s.node_path + bytes([b]), lead_space_ok=False)
        return s

    def is_complete(self, s: VenbaState) -> bool:
        if s.k != 14 or not s.node_path:
            return False
        n = self.node(s)
        return n is not None and n.value is not None

    def finished_words(self, s: VenbaState) -> List[str]:
        out = list(s.words)
        if self.is_complete(s):
            out.append(self.node(s).value.word)
        return out

    def monai_target(self, s: VenbaState) -> Optional[str]:
        return None if s.k in LINE_STARTS else s.line_monai


class VenbaLogitsProcessor:
    """Hugging Face logits processor wrapping :class:`VenbaDecoder`.

    Call :meth:`reset` (or build a new processor) before each ``generate``.
    """

    def __init__(self, tokenizer, decoder: VenbaDecoder, eos_token_id: Optional[int] = None):
        import torch  # noqa: F401  (fail early with a clear message if torch is missing)

        self.decoder = decoder
        self.token_bytes = token_bytes_table(tokenizer)
        self.eos = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        self._cache: Dict[VenbaState, Tuple[list, list, list]] = {}
        self.prompt_len: Optional[int] = None

    def reset(self):
        self.prompt_len = None

    def state_for(self, generated: Sequence[int]) -> Optional[VenbaState]:
        s = self.decoder.initial()
        for tid in generated:
            if tid == self.eos:
                break
            data = self.token_bytes[tid] if tid < len(self.token_bytes) else None
            if data is None:
                return None
            s = self.decoder.advance(s, data)
            if s is None:
                return None
        return s

    def allowed(self, s: VenbaState):
        hit = self._cache.get(s)
        if hit is not None:
            return hit
        allowed, boosted, pozhippu = [], [], []
        for tid, data in enumerate(self.token_bytes):
            if not data:
                continue
            ns = self.decoder.advance(s, data)
            if ns is None:
                continue
            allowed.append(tid)
            if ns.node_path and (ns.k > s.k or not s.node_path):
                target = self.decoder.monai_target(ns)
                n = self.decoder.node(ns)
                if target and n is not None and target in n.monai:
                    is_pozhippu = ns.k - self.decoder.lex.line_start(ns.k) == 2
                    (pozhippu if is_pozhippu else boosted).append(tid)
        if self.decoder.is_complete(s) and self.eos is not None:
            allowed.append(self.eos)
        self._cache[s] = (allowed, boosted, pozhippu)
        return self._cache[s]

    def __call__(self, input_ids, scores):
        import torch

        if self.prompt_len is None:
            self.prompt_len = input_ids.shape[1]
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:].tolist()
            s = self.state_for(gen)
            mask = torch.full_like(scores[b], float("-inf"))
            if s is None:
                if self.eos is not None:
                    mask[self.eos] = 0.0
            else:
                allowed, boosted, pozhippu = self.allowed(s)
                if not allowed and self.eos is not None:
                    allowed = [self.eos]
                mask[allowed] = 0.0
                if boosted:
                    mask[boosted] += self.decoder.monai_boost
                if pozhippu:
                    mask[pozhippu] += self.decoder.monai_boost * self.decoder.pozhippu_factor
            scores[b] = scores[b] + mask
        return scores


# ==============================================================================
# ICD-10 constrained code emission
# ==============================================================================


class ICD10CodeTrie:
    def __init__(self, codes: Iterable[str]):
        self.codes = sorted({c.strip().upper() for c in codes if c.strip()})
        self.trie = ByteTrie((c, c) for c in self.codes)

    def walk(self, prefix: str) -> Optional[TrieNode]:
        n = self.trie.root
        for b in prefix.encode("utf-8"):
            n = n.children.get(b)
            if n is None:
                return None
        return n

    def is_code(self, s: str) -> bool:
        n = self.walk(s)
        return n is not None and n.value is not None


class ICD10LogitsProcessor:
    """After ``marker`` appears in the generated text (or at the end of the
    prompt), restrict output to ``CODE[; CODE]*`` then newline / EOS, where every
    CODE is in ``allowed_codes``.  Duplicate codes are not allowed."""

    def __init__(self, tokenizer, allowed_codes: Iterable[str], marker: str = "CODES:", separator: str = ";"):
        import torch  # noqa: F401

        self.tok = tokenizer
        self.trie = ICD10CodeTrie(allowed_codes)
        self.marker = marker
        self.sep = separator
        self.eos = tokenizer.eos_token_id
        self.token_bytes = token_bytes_table(tokenizer)
        self.token_text = [b.decode("utf-8", errors="replace") if b else None for b in self.token_bytes]
        self._cache: Dict[Tuple[str, frozenset], List[int]] = {}

    # state = (partial code, codes already emitted); returns None if invalid
    def _advance(self, partial: str, done: frozenset, text: str):
        for ch in text:
            if ch == " ":
                if partial:
                    return None
                continue
            if ch in (self.sep, "\n"):
                if not self.trie.is_code(partial) or partial in done:
                    return None
                done, partial = done | {partial}, ""
                if ch == "\n":
                    return ("<END>", done)
                if not self._live("", done):        # no unused code left to follow the separator
                    return None
                continue
            partial += ch
            if not self._live(partial, done):
                return None
        return (partial, done)

    def _live(self, partial: str, done: frozenset) -> bool:
        """Can ``partial`` still be completed to a code that has not been emitted yet?"""
        return any(c.startswith(partial) and c not in done for c in self.trie.codes)

    def allowed(self, partial: str, done: frozenset) -> List[int]:
        key = (partial, done)
        if key in self._cache:
            return self._cache[key]
        out = [tid for tid, text in enumerate(self.token_text)
               if text and self._advance(partial, done, text) is not None]
        if self.trie.is_code(partial) and partial not in done and self.eos is not None:
            out.append(self.eos)
        self._cache[key] = out
        return out

    def parse_state(self, text: str):
        idx = text.rfind(self.marker)
        if idx < 0:
            return None
        body = text[idx + len(self.marker):]
        if "\n" in body:
            return "<END>", frozenset()
        parts = body.split(self.sep)
        done = frozenset(p.strip() for p in parts[:-1])
        return parts[-1].strip(), done

    def __call__(self, input_ids, scores):
        import torch

        for b in range(input_ids.shape[0]):
            text = self.tok.decode(input_ids[b], skip_special_tokens=True)
            st = self.parse_state(text)
            if st is None:
                continue
            partial, done = st
            mask = torch.full_like(scores[b], float("-inf"))
            if partial == "<END>":
                if self.eos is not None:
                    mask[self.eos] = 0.0
            else:
                ids = self.allowed(partial, done) or ([self.eos] if self.eos is not None else [])
                mask[ids] = 0.0
            scores[b] = scores[b] + mask
        return scores


def parse_codes(text: str, marker: str = "CODES:", separator: str = ";") -> List[str]:
    idx = text.rfind(marker)
    if idx < 0:
        return []
    body = text[idx + len(marker):].split("\n")[0]
    return [c.strip() for c in body.split(separator) if c.strip()]
