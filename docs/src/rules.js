// JavaScript port of yappu_fsm.rules.datalog (evaluation + provenance) and RuleCoder ordering.
// Kept in sync by tests/test_js_port.py::test_js_rules_match_python.
const RJ = (() => {
  const SEP = "\u0001", PATIENT = "patient";
  const key = (f) => f.join(SEP);

  function run(clauses, facts) {
    const F = new Map(), byPred = {}, why = new Map();
    const add = (f, w) => { const k = key(f); if (F.has(k)) return false; F.set(k, f); (byPred[f[0]] = byPred[f[0]] || []).push(f); why.set(k, w); return true; };
    facts.forEach((f) => add(f, ["fact", []]));
    const strata = [...new Set(clauses.map((c) => c.stratum))].sort((a, b) => a - b);
    function* match(body, i, b, used) {
      if (i === body.length) { yield [b, used]; return; }
      const l = body[i];
      if (l.neg) {
        const g = [l.pred, ...l.args.map((a) => (a && a.v ? b[a.v] : a))];
        if (!F.has(key(g))) yield* match(body, i + 1, b, [...used, ["not", ...g]]);
        return;
      }
      for (const f of (byPred[l.pred] || []).slice()) {
        if (f.length !== l.args.length + 1) continue;
        const nb = { ...b };
        let ok = true;
        for (let j = 0; j < l.args.length && ok; j++) {
          const a = l.args[j], v = f[j + 1];
          if (a && a.v) { if (a.v in nb && nb[a.v] !== v) ok = false; else nb[a.v] = v; }
          else if (a !== v) ok = false;
        }
        if (ok) yield* match(body, i + 1, nb, [...used, f]);
      }
    }
    for (const s of strata) {
      const cs = clauses.filter((c) => c.stratum === s);
      let changed = true;
      while (changed) {
        changed = false;
        for (const c of cs) for (const [b, used] of [...match(c.body, 0, {}, [])]) {
          const h = [c.head.pred, ...c.head.args.map((a) => (a && a.v ? b[a.v] : a))];
          if (add(h, [c.rid, used])) changed = true;
        }
      }
    }
    return { has: (f) => F.has(key(f)), query: (p) => byPred[p] || [], why: (f) => why.get(key(f)) || ["?", []] };
  }

  function explain(m, f, rulesById, depth = 0, seen = new Set()) {
    const pad = "  ".repeat(depth);
    if (f[0] === "not") return [{ depth, text: `not ${f[1]}(${f.slice(2).join(", ")})`, tag: "absent" }];
    const [rid, used] = m.why(f);
    const out = [{ depth, text: `${f[0]}(${f.slice(1).join(", ")})`, tag: rid === "fact" ? "given" : rid, src: rulesById[rid] || "" }];
    if (seen.has(key(f))) return out;
    seen.add(key(f));
    for (const u of used) if (u[0] !== "patient") out.push(...explain(m, u, rulesById, depth + 1, seen));
    return out;
  }

  function walk(m, f, seen = new Set()) {
    if (seen.has(key(f))) return [];
    seen.add(key(f));
    let out = [f];
    for (const u of m.why(f)[1]) if (u[0] !== "not") out = out.concat(walk(m, u, seen));
    return out;
  }

  const cmp = (a, b) => (a[0] - b[0]) || (a[1] - b[1]);
  function baseOrder(codes, findings, m) {
    const keyOf = new Map(), conceptPos = new Map();
    findings.forEach((f, i) => {
      const c = f.entity.replace(/ /g, "-");
      if (!conceptPos.has(c)) conceptPos.set(c, i);
      for (const code of f.codes) if (!keyOf.has(code)) keyOf.set(code, [i, keyOf.size]);
    });
    const given = new Set(keyOf.keys()), base = (rid) => rid.split(".")[0];
    for (const c of codes) {
      if (given.has(c)) continue;
      const w = m.why(["code", PATIENT, c]);
      const links = [];
      for (const bf of m.query("blocked")) {
        const bw = m.why(bf);
        if (keyOf.has(bf[2]) && (base(bw[0]) === base(w[0]) || bw[1].some((u) => key(u) === key(["code", PATIENT, c])))) links.push(keyOf.get(bf[2]));
      }
      if (links.length) { const mn = links.sort(cmp)[0]; keyOf.set(c, [mn[0], mn[1] - 0.5]); continue; }
      const pos = walk(m, ["code", PATIENT, c]).filter((u) => u.length > 2 && conceptPos.has(u[2])).map((u) => conceptPos.get(u[2]));
      keyOf.set(c, pos.length ? [Math.min(...pos), -1] : [1e6, keyOf.size]);
    }
    return codes.map((c, i) => [c, i]).sort((a, b) => cmp(keyOf.get(a[0]) || [1e6, 0], keyOf.get(b[0]) || [1e6, 0]) || a[1] - b[1]).map((x) => x[0]);
  }

  function sequence(codes, before) {
    const present = new Set(codes);
    const edges = before.filter(([a, b]) => present.has(a) && present.has(b) && a !== b);
    const out = [], rem = [...codes];
    while (rem.length) {
      const idx = rem.findIndex((c) => !edges.some(([a, b]) => b === c && rem.includes(a)));
      if (idx < 0) { out.push(...rem); break; }
      out.push(rem.splice(idx, 1)[0]);
    }
    return out;
  }

  function code(clauses, facts, findings, disabled = new Set()) {
    const active = clauses.filter((c) => !disabled.has(c.base));
    const m = run(active, facts);
    const finals = m.query("final").map((f) => f[2]);
    const codes = sequence(baseOrder(finals, findings, m), m.query("precedes").map((f) => [f[2], f[3]]));
    const blocked = m.query("blocked").map((f) => f[2]).filter((c) => m.has(["code", PATIENT, c]));
    const conflicts = m.query("conflict").map((f) => [f[2], f[3]]);
    return { m, codes, blocked: [...new Set(blocked)].sort(), conflicts };
  }
  return { run, explain, code };
})();
if (typeof module !== "undefined") module.exports = RJ;
