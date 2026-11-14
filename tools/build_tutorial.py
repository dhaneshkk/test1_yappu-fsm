"""Inline docs/src/prosody.js and docs/tutorial_data.json into docs/tutorial.html."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "docs"
html = (ROOT / "src" / "tutorial.template.html").read_text(encoding="utf-8")
js = (ROOT / "src" / "prosody.js").read_text(encoding="utf-8")
rules_js = (ROOT / "src" / "rules.js").read_text(encoding="utf-8")
vision_js = (ROOT / "src" / "vision.js").read_text(encoding="utf-8")
data = (ROOT / "tutorial_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
html = html.replace("/*__PROSODY_JS__*/", js).replace("/*__RULES_JS__*/", rules_js).replace("/*__VISION_JS__*/", vision_js).replace("/*__DATA_JSON__*/", data)
(ROOT / "tutorial.html").write_text(html, encoding="utf-8")
print(ROOT / "tutorial.html", len(html.encode()) // 1024, "KB")
