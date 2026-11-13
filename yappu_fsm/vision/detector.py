"""YOLOX-Nano object detection on monochrome frames with onnxruntime (no PyTorch).

Model: the official ``yolox_nano.onnx`` export (input ``images`` 1x3x416x416, output
``output`` 1x3549x85 = 52²+26²+13² anchor points x [cx, cy, w, h, obj, 80 COCO classes]).

Monochrome: each frame is converted to grey (optionally contrast-equalised with CLAHE for
low light) and the single channel is replicated to the three channels YOLOX expects. COCO
weights were trained on colour images, so expect somewhat lower recall on grey input;
fine-tune on grey frames for production.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

YOLOX_NANO_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass
class Detection:
    box: Tuple[float, float, float, float]   # x1, y1, x2, y2 in frame pixels
    score: float
    cls: int

    @property
    def label(self) -> str:
        return COCO_CLASSES[self.cls]


def to_monochrome(frame: np.ndarray, clahe: bool = False) -> np.ndarray:
    """BGR/BGRA/grey frame -> single-channel uint8 grey."""
    import cv2

    if frame.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        frame = cv2.cvtColor(frame, code)
    if clahe:
        frame = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(frame)
    return frame


def letterbox(gray: np.ndarray, size: Tuple[int, int] = (416, 416)) -> Tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, pad with 114 (YOLOX convention), return CHW float32 + ratio."""
    import cv2

    h, w = gray.shape[:2]
    r = min(size[0] / h, size[1] / w)
    resized = cv2.resize(gray, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full(size, 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    chw = np.repeat(canvas[None, :, :], 3, axis=0).astype(np.float32)   # grey -> 3 identical channels
    return chw, r


def decode(raw: np.ndarray, size: Tuple[int, int] = (416, 416), strides: Sequence[int] = (8, 16, 32)) -> np.ndarray:
    """YOLOX head output (N, 85) -> boxes in letterboxed pixels (cx, cy, w, h, ...)."""
    grids, exp_strides = [], []
    for s in strides:
        hs, ws = size[0] // s, size[1] // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        exp_strides.append(np.full((hs * ws, 1), s))
    grids = np.concatenate(grids, 0)
    exp_strides = np.concatenate(exp_strides, 0)
    out = raw.copy()
    out[:, :2] = (out[:, :2] + grids) * exp_strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * exp_strides
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def postprocess(raw: np.ndarray, ratio: float, size=(416, 416), score_thr: float = 0.1, nms_thr: float = 0.45,
                classes: Optional[Sequence[int]] = None) -> List[Detection]:
    """Decode, score = objectness x class probability, class-wise NMS, map back to frame pixels.

    ``score_thr`` is kept low on purpose: the tracker's second association stage
    (ByteTrack/BoT-SORT) needs the low-confidence boxes."""
    p = decode(raw, size)
    boxes = np.empty((len(p), 4), dtype=np.float32)
    boxes[:, 0] = p[:, 0] - p[:, 2] / 2
    boxes[:, 1] = p[:, 1] - p[:, 3] / 2
    boxes[:, 2] = p[:, 0] + p[:, 2] / 2
    boxes[:, 3] = p[:, 1] + p[:, 3] / 2
    boxes /= ratio
    cls_scores = p[:, 4:5] * p[:, 5:]
    cls = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(cls)), cls]
    mask = score >= score_thr
    if classes is not None:
        mask &= np.isin(cls, list(classes))
    dets: List[Detection] = []
    for c in np.unique(cls[mask]):
        idx = np.where(mask & (cls == c))[0]
        for k in nms(boxes[idx], score[idx], nms_thr):
            i = idx[k]
            dets.append(Detection(tuple(float(v) for v in boxes[i]), float(score[i]), int(c)))
    return sorted(dets, key=lambda d: -d.score)


class YoloxNano:
    """onnxruntime wrapper. ``detect(frame)`` accepts colour or grey frames and always runs on grey."""

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (416, 416), score_thr: float = 0.1,
                 nms_thr: float = 0.45, classes: Optional[Sequence[str]] = None, clahe: bool = False,
                 providers: Optional[Sequence[str]] = None):
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found; download it with:\n  curl -L -o {model_path} {YOLOX_NANO_URL}")
        self.session = ort.InferenceSession(model_path, providers=list(providers or ["CPUExecutionProvider"]))
        self.input_name = self.session.get_inputs()[0].name
        self.size, self.score_thr, self.nms_thr, self.clahe = input_size, score_thr, nms_thr, clahe
        self.classes = None if classes is None else [COCO_CLASSES.index(c) for c in classes]

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = to_monochrome(frame, self.clahe)
        blob, ratio = letterbox(gray, self.size)
        raw = self.session.run(None, {self.input_name: blob[None]})[0][0]
        return postprocess(raw, ratio, self.size, self.score_thr, self.nms_thr, self.classes)
