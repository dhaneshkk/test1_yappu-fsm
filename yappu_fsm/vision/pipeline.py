"""Monochrome camera -> YOLOX-Nano -> BoT-SORT-style tracker -> rule-based alerts.

Run on a laptop camera::

    pip install -e ".[vision]"
    yappu vision --download                      # fetch yolox_nano.onnx once (3.7 MB)
    yappu vision --source 0 --zone restricted=0,0,0.4,1

Keys in the preview window: q = quit.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .detector import COCO_CLASSES, YOLOX_NANO_URL, YoloxNano, to_monochrome
from .events import AlertEvent, EventEngine, Zone, parse_vision_rules, parse_zone
from .tracker import LOST, BotSortLite

DEFAULT_MODEL = Path(os.environ.get("YAPPU_YOLOX_NANO", Path.home() / ".cache" / "yappu" / "yolox_nano.onnx"))
DEFAULT_CLASSES = ("person", "cell phone", "cup", "bottle", "laptop", "book")


def download_model(path: Path = DEFAULT_MODEL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"downloading {YOLOX_NANO_URL} -> {path}")
        urllib.request.urlretrieve(YOLOX_NANO_URL, path)
    return path


@dataclass
class FrameResult:
    gray: np.ndarray
    detections: list
    tracks: list
    events: List[AlertEvent]
    ms: dict


class VisionPipeline:
    def __init__(self, detector: YoloxNano, tracker: BotSortLite, engine: EventEngine):
        self.detector, self.tracker, self.engine = detector, tracker, engine

    def process(self, frame: np.ndarray, t: float) -> FrameResult:
        t0 = time.perf_counter()
        gray = to_monochrome(frame, self.detector.clahe)
        dets = self.detector.detect(gray)
        t1 = time.perf_counter()
        self.tracker.update(dets, t, gray)
        tracks = self.tracker.active(include_lost=self.engine.include_lost)
        tracks = [tr for tr in tracks if tr.state != LOST or t - tr.last_time <= self.engine.timer.grace]
        t2 = time.perf_counter()
        events = self.engine.step(tracks, t, (gray.shape[1], gray.shape[0]))
        t3 = time.perf_counter()
        return FrameResult(gray, dets, tracks, events,
                           {"detect": (t1 - t0) * 1e3, "track": (t2 - t1) * 1e3, "rules": (t3 - t2) * 1e3})


def draw(res: FrameResult, zones: Sequence[Zone], active) -> np.ndarray:
    """Grey frame with overlays (kept monochrome; alerts in white on black)."""
    import cv2

    img = cv2.cvtColor(res.gray, cv2.COLOR_GRAY2BGR)
    h, w = res.gray.shape
    for z in zones:
        x1, y1, x2, y2 = int(z.box[0] * w), int(z.box[1] * h), int(z.box[2] * w), int(z.box[3] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(img, z.name, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    alerted = {s for s, _ in active}
    for tr in res.tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        thick = 3 if str(tr.tid) in alerted else 1
        shade = (255, 255, 255) if tr.state != LOST else (120, 120, 120)
        cv2.rectangle(img, (x1, y1), (x2, y2), shade, thick)
        cv2.putText(img, f"#{tr.tid} {COCO_CLASSES[tr.cls]}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, shade, 1)
    for i, (s, e) in enumerate(sorted(active)):
        txt = f"ALERT {e} ({'#' + s if s != 'scene' else s})"
        cv2.rectangle(img, (0, 20 * i), (12 + 8 * len(txt), 20 * i + 18), (0, 0, 0), -1)
        cv2.putText(img, txt, (6, 20 * i + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def run(source="0", model: Optional[str] = None, zones: Sequence[str] = ("restricted=0,0,0.4,1",),
        rules: Optional[str] = None, classes: Sequence[str] = DEFAULT_CLASSES, clahe: bool = False,
        show: bool = True, max_frames: Optional[int] = None, gmc: bool = False, log: Optional[str] = None,
        save: Optional[str] = None, quiet: bool = False) -> List[AlertEvent]:
    import cv2

    from .. import data_path

    zs = [parse_zone(z) for z in zones]
    rule_text = Path(rules or data_path("vision_rules.ace")).read_text(encoding="utf-8")
    rs = parse_vision_rules(rule_text)
    used = {a for r in rs for k, a, _ in r.durations if k == "in_zone"} | {z for r in rs for z, _ in r.crowds}
    rs = [r for r in rs if all(a in {z.name for z in zs} for k, a, _ in r.durations if k == "in_zone")
          and all(z in {x.name for x in zs} for z, _ in r.crowds)]
    if not quiet and used - {z.name for z in zs}:
        print(f"note: rules for zones {sorted(used - {z.name for z in zs})} skipped (zone not defined)")
    cam = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {source!r} (camera permission? another app using it?)")
    file_fps = cap.get(cv2.CAP_PROP_FPS) if not isinstance(cam, int) else 0
    det = YoloxNano(str(model or DEFAULT_MODEL), classes=list(classes), clahe=clahe)
    trk = BotSortLite(frame_rate=file_fps or 30, gmc=gmc)
    eng = EventEngine(rs, zs)
    pipe = VisionPipeline(det, trk, eng)
    writer, logf, all_events, n = None, open(log, "a") if log else None, [], 0
    t_start = time.monotonic()
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / file_fps if file_fps else time.monotonic() - t_start
            res = pipe.process(frame, t)
            n += 1
            for ev in res.events:
                all_events.append(ev)
                if not quiet:
                    print(f"[{t:7.2f}s] {ev.kind.upper()} {ev.event} ({'track #' + ev.subject if ev.subject != 'scene' else 'scene'})")
                    if ev.kind == "raised":
                        print("    " + "\n    ".join(ev.proof))
                if logf:
                    logf.write(json.dumps({"t": t, **ev.__dict__}) + "\n")
            if show or save:
                img = draw(res, zs, eng.alerts.active())
                ms = res.ms
                cv2.putText(img, f"det {ms['detect']:.0f}ms trk {ms['track']:.1f}ms rules {ms['rules']:.1f}ms",
                            (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                if save:
                    if writer is None:
                        writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), file_fps or 15,
                                                 (img.shape[1], img.shape[0]))
                    writer.write(img)
                if show:
                    cv2.imshow("yappu vision (monochrome)", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer:
            writer.release()
        if logf:
            logf.close()
        if show:
            cv2.destroyAllWindows()
    return all_events
