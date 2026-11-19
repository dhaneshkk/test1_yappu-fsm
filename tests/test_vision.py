"""Vision layer: YOLOX decode, BoT-SORT-style tracking, temporal rules, alerts, MCC."""
import os
import random
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")      # vision extra; the core CI job has no numpy
pytest.importorskip("cv2")
pytest.importorskip("scipy")

from yappu_fsm import data_path  # noqa: E402
from yappu_fsm.clinical import mcc  # noqa: E402
from yappu_fsm.vision import (CONFIRMED, LOST, BotSortLite, Detection, EventEngine, VisionRuleError,  # noqa: E402
                              decode, letterbox, parse_vision_rule, parse_vision_rules, parse_zone, postprocess,
                              to_monochrome)

PERSON, PHONE = 0, 67
MODEL = Path(os.environ.get("YAPPU_YOLOX_NANO", Path.home() / ".cache" / "yappu" / "yolox_nano.onnx"))


def det(x, y, w=40, h=100, score=0.9, cls=PERSON):
    return Detection((x, y, x + w, y + h), score, cls)


# ---------------------------------------------------------------- detector

def test_monochrome_letterbox():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gray = to_monochrome(frame)
    assert gray.ndim == 2
    blob, r = letterbox(gray)
    assert blob.shape == (3, 416, 416) and r == pytest.approx(416 / 640)
    assert np.array_equal(blob[0], blob[1]) and np.array_equal(blob[1], blob[2])   # grey replicated
    assert (blob[:, 320:, :] == 114).all()                                          # YOLOX padding value


def test_decode_and_postprocess_known_box():
    raw = np.full((3549, 85), -10.0, dtype=np.float32)
    raw[:, 4:] = 0.0
    # anchor 0 of stride 8 is grid (0, 0); put a 64x128 box centred at (100, 60) there
    i = 0
    raw[i, 0], raw[i, 1] = 100 / 8, 60 / 8
    raw[i, 2], raw[i, 3] = np.log(64 / 8), np.log(128 / 8)
    raw[i, 4], raw[i, 5 + PERSON] = 0.9, 1.0
    p = decode(raw)
    assert p[i, :4].tolist() == pytest.approx([100, 60, 64, 128])
    dets = postprocess(raw, ratio=0.5, score_thr=0.5)
    assert len(dets) == 1 and dets[0].label == "person"
    assert dets[0].box == pytest.approx((136, -8, 264, 248))   # divided by the letterbox ratio


@pytest.mark.skipif(not MODEL.exists(), reason="yolox_nano.onnx not downloaded")
def test_real_model_runs_on_monochrome():
    from yappu_fsm.vision import YoloxNano
    m = YoloxNano(str(MODEL))
    blank = np.full((480, 640), 30, dtype=np.uint8)
    assert m.detect(blank) == [] or all(d.score < 0.5 for d in m.detect(blank))


# ---------------------------------------------------------------- tracker

def run_tracker(frames, **kw):
    tr = BotSortLite(frame_rate=30, **kw)
    ids = []
    for i, ds in enumerate(frames):
        tr.update(ds, i / 30)
        ids.append({round(t.box[0]): t.tid for t in tr.active()})
    return tr, ids


def test_crossing_keeps_identities():
    frames = [[det(100 + 6 * i, 100), det(400 - 6 * i, 110)] for i in range(50)]
    tr, _ = run_tracker(frames)
    a, b = sorted(tr.active(), key=lambda t: t.box[0])
    assert {a.tid, b.tid} == {1, 2}
    assert a.box[0] < b.box[0] and a.tid == 2 and b.tid == 1   # they crossed and kept their ids


def test_occlusion_gap_recovered():
    frames = [[det(100 + 3 * i, 100)] if not 20 <= i < 28 else [] for i in range(40)]
    tr, _ = run_tracker(frames)
    (t,) = tr.active()
    assert t.tid == 1 and any("LOST" in h for h in t.history) and t.state == CONFIRMED


def test_low_score_frames_recovered_by_second_stage():
    frames = [[det(100 + 2 * i, 100, score=0.9 if i % 3 else 0.3)] for i in range(30)]
    tr, _ = run_tracker(frames)
    (t,) = tr.active()
    assert t.tid == 1 and not any("LOST" in h for h in t.history[2:])


def test_single_frame_false_positive_never_confirmed():
    frames = [[det(100, 100)] + ([det(500, 300, cls=PHONE)] if i == 5 else []) for i in range(20)]
    tr, _ = run_tracker(frames)
    assert [t.tid for t in tr.active()] == [1]


def test_lost_track_removed_after_buffer():
    frames = [[det(100, 100)]] * 5 + [[]] * 40
    tr, _ = run_tracker(frames, track_buffer=30)
    assert tr.active(include_lost=True) == [] and tr.removed


def test_class_aware_association():
    frames = [[det(100, 100)]] * 5 + [[det(100, 100, cls=PHONE)]] * 5
    tr, _ = run_tracker(frames)
    assert all(t.cls == PHONE for t in tr.active())


# ---------------------------------------------------------------- rules and events

ZONE = parse_zone("restricted=0,0,0.5,1")


class T:  # minimal track stand-in for the event engine
    def __init__(self, tid, box, cls=PERSON):
        self.tid, self.box, self.cls, self.state, self.last_time = tid, box, cls, CONFIRMED, 0


def engine(text=None, **kw):
    rules = parse_vision_rules(text or open(data_path("vision_rules.ace"), encoding="utf-8").read())
    return EventEngine(rules, [ZONE], **kw)


def feed(eng, timeline, fps=10):
    events = []
    for i, tracks in enumerate(timeline):
        events += [(round(i / fps, 2), e.kind, e.subject, e.event) for e in eng.step(tracks, i / fps, (1000, 500))]
    return events


IN, OUT = (100, 100, 200, 300), (700, 100, 800, 300)


def test_zone_intrusion_after_two_seconds_once():
    ev = feed(engine(), [[T(1, IN)]] * 40)
    assert ev == [(2.0, "raised", "1", "zone-intrusion")]


def test_short_gap_does_not_reset_timer_but_long_gap_does():
    short = [[T(1, IN)]] * 12 + [[]] * 3 + [[T(1, IN)]] * 10        # 0.3 s gap < 0.7 s grace
    assert ("raised", "zone-intrusion") in [(k, e) for _, k, _, e in feed(engine(), short)]
    long = [[T(1, IN)]] * 15 + [[]] * 12 + [[T(1, IN)]] * 25         # 1.2 s gap resets the timer
    raised = [t for t, k, _, e in feed(engine(), long) if k == "raised" and e == "zone-intrusion"]
    assert raised == [4.7]                                            # 2 s after re-entry at 2.7 s


def test_alert_clears_after_cooldown():
    ev = feed(engine(cooldown=1.0), [[T(1, IN)]] * 25 + [[T(1, OUT)]] * 30)
    kinds = [(k, e) for _, k, _, e in ev]
    assert kinds == [("raised", "zone-intrusion"), ("cleared", "zone-intrusion")]


def test_phone_use_by_association():
    phone = T(9, (720, 150, 740, 190), cls=PHONE)
    ev = feed(engine(), [[T(1, OUT), phone]] * 40)
    assert ("raised", "1", "phone-use") in [(k, s, e) for _, k, s, e in ev]
    assert not any(e == "phone-use" for *_, e in feed(engine(), [[T(1, OUT)]] * 40))


def test_crowd_scene_alert_and_proof():
    eng = engine()
    ev = eng.step([T(1, IN), T(2, (300, 100, 350, 300))], 0.0, (1000, 500))
    assert [(e.subject, e.event) for e in ev] == [("scene", "crowding")]
    assert "If at least 2 persons are in the restricted-zone" in ev[0].proof[0]


def test_rule_errors():
    with pytest.raises(VisionRuleError, match="not a COCO class"):
        parse_vision_rule("If the person T holds a helmet for at least 2 seconds then a ppe alert is raised for the person T.")
    with pytest.raises(VisionRuleError, match="must be introduced"):
        parse_vision_rule("If the person T does not hold a cell-phone then a x alert is raised for the person T.")
    with pytest.raises(VisionRuleError, match="no zone named"):
        EventEngine(parse_vision_rules("If a person T is in the loading-zone for at least 1 second "
                                       "then a z alert is raised for the person T."), [ZONE])
    with pytest.raises(VisionRuleError, match="not in the vision rule language"):
        parse_vision_rule("If a person T runs then a z alert is raised.")


# ---------------------------------------------------------------- MCC: temporal rules vs per-frame rule

def test_temporal_rules_beat_per_frame_alerting_on_flicker():
    """Simulated monochrome camera: the detector misses 25 % of frames and a passer-by
    crosses the zone for 0.8 s. Gold: 'intrusion' only while someone has truly stayed >= 2 s."""
    rng = random.Random(3)
    fps, frames, gold, timeline = 10, 300, [], []
    for i in range(frames):
        t = i / fps
        stay = 5 <= t < 17                       # person 1 stays in the zone 12 s
        passer = 22 <= t < 22.8                  # person 2 walks through
        tracks = []
        if stay and rng.random() > 0.25:
            tracks.append(T(1, IN))
        if not stay:
            tracks.append(T(1, OUT))
        if passer and rng.random() > 0.25:
            tracks.append(T(2, IN))
        timeline.append(tracks)
        gold.append(str(7 <= t < 17))
    eng = engine("INTRUSION: If a person T is in the restricted-zone for at least 2 seconds "
                 "then a zone-intrusion alert is raised for the person T.", cooldown=0.0)
    pred_rules, pred_naive = [], []
    for i, tracks in enumerate(timeline):
        eng.step(tracks, i / fps, (1000, 500))
        pred_rules.append(str(bool(eng.alerts.active())))
        pred_naive.append(str(any(tr.box[0] < 500 for tr in tracks)))   # per-frame: anyone in zone now
    m_rules, m_naive = mcc(gold, pred_rules), mcc(gold, pred_naive)
    assert m_rules > 0.9 and m_rules > m_naive + 0.1, (m_rules, m_naive)
