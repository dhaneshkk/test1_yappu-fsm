"""Inline docs/src/prosody.js and docs/tutorial_data.json into docs/tutorial.html."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "docs"
html = (ROOT / "src" / "tutorial.template.html").read_text(encoding="utf-8")
js = (ROOT / "src" / "prosody.js").read_text(encoding="utf-8")
data = (ROOT / "tutorial_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
html = html.replace("/*__PROSODY_JS__*/", js).replace("/*__DATA_JSON__*/", data)
(ROOT / "tutorial.html").write_text(html, encoding="utf-8")
print(ROOT / "tutorial.html", len(html.encode()) // 1024, "KB")
