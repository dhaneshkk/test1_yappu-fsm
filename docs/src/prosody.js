// JavaScript port of yappu_fsm.tamil + yappu_fsm.prosody (kept in sync by tests/test_js_port.py)
const Y = (() => {
  const PULLI = "்", AU = "ௗ", AYTHAM = "ஃ";
  const NEDIL_SIGNS = new Set(["ா", "ீ", "ூ", "ே", "ை", "ோ", "ௌ"]);
  const KURIL_SIGNS = new Set(["ி", "ு", "ெ", "ொ"]);
  const COMBINING = new Set([...NEDIL_SIGNS, ...KURIL_SIGNS, PULLI, AU]);
  const IK = new Set(["அ", "இ", "உ", "எ", "ஒ"]), IN = new Set(["ஆ", "ஈ", "ஊ", "ஏ", "ஐ", "ஓ", "ஔ"]);
  const VALLINAM = new Set(["க", "ச", "ட", "த", "ப", "ற"]);
  const isTa = (c) => c >= "஀" && c <= "௿";
  const isCons = (c) => c >= "க" && c <= "ஹ";

  function graphemes(word) {
    const out = [];
    for (const ch of word.normalize("NFC")) {
      if (COMBINING.has(ch) && out.length) out[out.length - 1] += ch;
      else if (isTa(ch)) out.push(ch);
    }
    return out;
  }
  function letterClass(g) {
    const b = g[0];
    if (IK.has(b)) return "K";
    if (IN.has(b)) return "N";
    if (b === AYTHAM) return "O";
    if (isCons(b)) {
      const s = g.slice(1, 2);
      if (s === PULLI) return "O";
      if (NEDIL_SIGNS.has(s)) return "N";
      return "K";
    }
    return null;
  }
  // mathirai (beats) per letter
  const BEATS = { K: 1, N: 2, O: 0.5 };

  function asaiUnits(word) {
    const gs = graphemes(word), cls = gs.map(letterClass), units = [];
    let i = 0;
    while (i < gs.length) {
      const c = cls[i];
      if (c === "O") { if (units.length) units[units.length - 1].text += gs[i]; i++; continue; }
      let text, kind;
      if (c === "K" && i + 1 < gs.length && (cls[i + 1] === "K" || cls[i + 1] === "N")) { text = gs[i] + gs[i + 1]; kind = "0"; i += 2; }
      else { text = gs[i]; kind = "1"; i += 1; }
      while (i < gs.length && cls[i] === "O") { text += gs[i]; i++; }
      units.push({ text, kind });
    }
    return units;
  }
  const encode = (w) => asaiUnits(w).map((u) => u.kind).join("");
  const split = (w) => asaiUnits(w).map((u) => u.text).join("/");

  const SEER = {
    "11": ["Thema", "Maacheer"], "01": ["Pulima", "Maacheer"], "00": ["Karuvilam", "Vilacheer"], "10": ["Koovilam", "Vilacheer"],
    "111": ["Themangai", "Kaycheer"], "011": ["Pulimangai", "Kaycheer"], "001": ["Karuvilangai", "Kaycheer"], "101": ["Koovilangai", "Kaycheer"],
    "110": ["Themangani", "Kanicheer"], "010": ["Pulimangani", "Kanicheer"], "000": ["Karuvilangani", "Kanicheer"], "100": ["Koovilangani", "Kanicheer"],
  };
  const classify = (a) => SEER[a] || [null, null];
  function kutri(word) {
    const gs = graphemes(word);
    if (gs.length < 2) return false;
    const l = gs[gs.length - 1];
    return [...l].length === 2 && VALLINAM.has(l[0]) && l[1] === "ு";
  }
  function eetrucheer(word) {
    const a = encode(word);
    if (a === "1") return "Naal";
    if (a === "0") return "Malar";
    if (kutri(word)) { if (a === "11") return "Kaasu"; if (a === "01") return "Pirappu"; }
    return null;
  }
  function expectedNext(a) {
    const c = classify(a)[1];
    return c === "Maacheer" ? "0" : (c === "Vilacheer" || c === "Kaycheer") ? "1" : null;
  }
  function thalai(prev, next) {
    const c = classify(prev)[1], s = next[0];
    if (c === "Maacheer") return s === "0" ? "Iyarseer Vendalai" : "Nerondru Aasiriyathalai";
    if (c === "Vilacheer") return s === "1" ? "Iyarseer Vendalai" : "Niraiondru Aasiriyathalai";
    if (c === "Kaycheer") return s === "1" ? "Venseer Vendalai" : "Kalithalai";
    if (c === "Kanicheer") return s === "0" ? "Ondriya Vanjithalai" : "Ondraadha Vanjithalai";
    return "Unknown";
  }
  const isVendalai = (t) => t === "Iyarseer Vendalai" || t === "Venseer Vendalai";
  function ethukai(word, strict = false) {
    const gs = graphemes(word);
    if (gs.length < 2) return null;
    return (letterClass(gs[0]) === "N" ? "N" : "K") + ":" + (strict ? gs[1] : gs[1][0]);
  }
  const VG = { "அ": "A", "ஆ": "A", "ஐ": "A", "ஔ": "A", "இ": "I", "ஈ": "I", "எ": "I", "ஏ": "I", "உ": "U", "ஊ": "U", "ஒ": "U", "ஓ": "U" };
  const SG = { "": "A", "ா": "A", "ை": "A", "ௌ": "A", "ி": "I", "ீ": "I", "ெ": "I", "ே": "I", "ு": "U", "ூ": "U", "ொ": "U", "ோ": "U" };
  const CG = { "ச": "சத", "த": "சத", "ஞ": "ஞந", "ந": "ஞந", "ம": "மவ", "வ": "மவ" };
  function monai(word) {
    const gs = graphemes(word);
    if (!gs.length) return null;
    const f = gs[0];
    if (VG[f[0]]) return "V:" + VG[f[0]];
    if (isCons(f[0])) return "C:" + (CG[f[0]] || f[0]) + ":" + (SG[f.slice(1, 2)] || "A");
    return null;
  }
  const parse = (text) => text.split(/[\s\-–—.,;!?]+/).filter(Boolean);

  function validate(poem, strict = false) {
    const words = Array.isArray(poem) ? poem : parse(poem);
    const rep = { valid: true, feet: [], errors: [], vikarpam: null, words };
    if (words.length !== 15) { rep.valid = false; rep.errors.push(`Expected 15 feet (4+4+4+3), got ${words.length}.`); return rep; }
    let prev = null;
    words.forEach((w, i) => {
      const asai = encode(w);
      let ok = true, msg = "", vp, cat;
      if (i === 14) { vp = eetrucheer(w); cat = vp ? "Eetrucheer" : null; if (!vp) { ok = false; msg = "final foot must be Naal/Malar/Kaasu/Pirappu"; } }
      else { [vp, cat] = classify(asai); if (!["Maacheer", "Vilacheer", "Kaycheer"].includes(cat)) { ok = false; msg = `${vp || asai} (${cat || "unclassified"}) not allowed in Venba`; } }
      let th = null;
      if (prev !== null) { th = thalai(prev, asai); if (!isVendalai(th)) { ok = false; msg = (msg ? msg + "; " : "") + "link is " + th; } }
      rep.feet.push({ index: i, word: w, asai, split: split(w), vaipadu: vp, category: cat, thalai: th, ok, msg });
      rep.valid = rep.valid && ok;
      prev = asai;
    });
    const a = [0, 4, 7].map((i) => ethukai(words[i], strict)), b = [8, 12].map((i) => ethukai(words[i], strict));
    if (a.includes(null) || new Set(a).size !== 1) { rep.valid = false; rep.errors.push(`Ethukai A mismatch across feet 1, 5, 8: ${a.join(", ")}`); }
    if (b.includes(null) || new Set(b).size !== 1) { rep.valid = false; rep.errors.push(`Ethukai B mismatch across feet 9, 13: ${b.join(", ")}`); }
    if (rep.valid) rep.vikarpam = a[0] === b[0] ? "Oru Vikarpam" : "Iru Vikarpam";
    return rep;
  }
  return { graphemes, letterClass, BEATS, asaiUnits, encode, split, SEER, classify, eetrucheer, expectedNext, thalai, isVendalai, ethukai, monai, parse, validate };
})();
if (typeof module !== "undefined") module.exports = Y;
