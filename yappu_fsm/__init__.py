"""yappu-fsm: Tamil prosody (Yappu Ilakkanam) as finite-state machines, constrained
decoding for Hugging Face models, and a rule-based clinical ICD-10 linker."""

from importlib import resources as _res

from .lexicon import VenbaLexicon
from .prosody import format_venba, validate_nerisai_venba
from .tamil import encode, encode_asai, ethukai_key, graphemes, monai_key, split_display

__version__ = "0.3.0"


def data_path(name: str) -> str:
    """Path to a bundled data file (e.g. 'icd10cm_example_lexicon.json')."""
    return str(_res.files(__package__) / "data" / name)


__all__ = ["VenbaLexicon", "format_venba", "validate_nerisai_venba", "encode", "encode_asai",
           "ethukai_key", "graphemes", "monai_key", "split_display", "data_path", "__version__"]
