"""Monochrome YOLOX-Nano detection, BoT-SORT-style tracking and rule-based alerts with proofs."""
from .detector import COCO_CLASSES, Detection, YoloxNano, decode, letterbox, postprocess, to_monochrome
from .events import AlertEvent, AlertMachine, EventEngine, Timer, VisionRuleError, Zone, parse_vision_rule, parse_vision_rules, parse_zone
from .tracker import CONFIRMED, LOST, REMOVED, TENTATIVE, BotSortLite, KalmanXYWH

__all__ = ["COCO_CLASSES", "Detection", "YoloxNano", "decode", "letterbox", "postprocess", "to_monochrome",
           "AlertEvent", "AlertMachine", "EventEngine", "Timer", "VisionRuleError", "Zone", "parse_vision_rule",
           "parse_vision_rules", "parse_zone", "CONFIRMED", "LOST", "REMOVED", "TENTATIVE", "BotSortLite", "KalmanXYWH"]
