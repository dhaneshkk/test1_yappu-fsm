// JavaScript port of yappu_fsm.vision.events.Timer and AlertMachine (kept in sync by
// tests/test_js_port.py::test_js_timer_matches_python).
const VJ = (() => {
  class Timer {
    constructor(grace) { this.grace = grace; this.start = new Map(); this.last = new Map(); }
    observe(k, on, t) {
      if (on) { if (!this.start.has(k)) this.start.set(k, t); this.last.set(k, t); }
      else if (this.last.has(k) && t - this.last.get(k) > this.grace) { this.start.delete(k); this.last.delete(k); }
    }
    duration(k, t) {
      if (!this.start.has(k)) return 0;
      return t - this.last.get(k) <= this.grace ? this.last.get(k) - this.start.get(k) : 0;
    }
  }
  class AlertMachine {
    constructor(cooldown) { this.cooldown = cooldown; this.state = new Map(); }
    step(derived, t) {
      const out = [];
      for (const k of derived) { const st = this.state.has(k) ? this.state.get(k)[0] : "IDLE"; if (st === "IDLE") out.push(["raised", k, t]); this.state.set(k, ["ACTIVE", t]); }
      for (const [k, [st, since]] of [...this.state]) {
        if (derived.has(k)) continue;
        if (st === "ACTIVE") this.state.set(k, ["COOLING", t]);
        else if (st === "COOLING" && t - since >= this.cooldown) { this.state.delete(k); out.push(["cleared", k, t]); }
      }
      return out;
    }
    active() { return [...this.state].filter(([, [st]]) => st === "ACTIVE").map(([k]) => k); }
  }
  // seen: booleans per frame -> {timer durations, alert active flags}
  function simulate(seen, fps, grace, thr, cooldown = 0) {
    const timer = new Timer(grace), am = new AlertMachine(cooldown), dur = [], active = [];
    seen.forEach((s, i) => {
      const t = i / fps;
      timer.observe("p", s, t);
      const d = timer.duration("p", t);
      dur.push(d);
      am.step(d >= thr ? new Set(["p"]) : new Set(), t);
      active.push(am.active().length > 0);
    });
    return { dur, active };
  }
  return { Timer, AlertMachine, simulate };
})();
if (typeof module !== "undefined") module.exports = VJ;
