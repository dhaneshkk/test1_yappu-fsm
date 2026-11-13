import pytest

from yappu_fsm import data_path
from yappu_fsm.clinical import ClinicalLinker, load_lexicon

POEM_1 = """எண்ணும் எழுத்தும் இசையோடு கூடினால்
பண்ணும் மணியாகும் பார்ப்பவர் - கண்ணும்
அமுதத் தமிழில் அமைந்த இலக்கணம்
நம்மொழி யாக்கும் இசை."""
POEM_2 = """வண்ண நிலவொளி வானில் தவழ்ந்திட
எண்ணற்ற விண்மீன்கள் ஏற்றிடும் - விண்ணில்
இருளினை நீக்கும் இனிய ஒளியால்
மருளினை மாற்றும் மதி."""
NANNERI = """அடக்க முடையா ரறிவிலரென் றெண்ணிக்
கடக்கக் கருதவும் வேண்டா - மடைத்தலையில்
ஓடுமீ னோட உறுமீன் வருமளவும்
வாடி யிருக்குமாம் கொக்கு."""


@pytest.fixture(scope="session")
def lexicon():
    return load_lexicon(data_path("icd10cm_example_lexicon.json"))


@pytest.fixture
def linker(lexicon):
    return ClinicalLinker(lexicon)
