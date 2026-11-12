"""requirements-api.txt must stay a subset of requirements.txt with identical pins."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pins(name):
    lines = (ROOT / name).read_text().splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}


def test_api_requirements_match_main_requirements():
    api, main = pins("requirements-api.txt"), pins("requirements.txt")
    assert api <= main, f"not in requirements.txt: {sorted(api - main)}"
    assert {p for p in main - api} == {p for p in main if p.startswith("streamlit")}
