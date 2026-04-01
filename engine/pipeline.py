# engine/pipeline.py
import cv2
import numpy as np
from ultralytics import YOLO
from engine.tracker_bridge import TrackerBridge
from engine.depth_fusion import extract_depth
from dataclasses import dataclass, field

@dataclass
class TrackedObject:
    track_id:  int
    class_id:  int
    bbox:      list        # [x1, y1, x2, y2]
    depth_m:   float       # 깊이 (미터), -1이면 감지 실패
    conf:      float
    velocity:  float = 0.0 # depth 변화량 (m/frame), 음수 = 접근 중

class Pipeline:
    def __init__(self, model_path: str):
        self.model   = YOLO(model_path)
        self.tracker = TrackerBridge()
        self.prev_depths: dict[int, float] = {}  # track_id → 이전 depth

    def process(self,
                frame: np.ndarray,
                depth_map: np.ndarray | None = None) -> list[TrackedObject]:
        """
        frame    : BGR numpy array (카메라 프레임)
        depth_map: ARCore depth map (mm, uint16). None이면 depth 없이 동작
        """
        # 1. YOLO 탐지
        yolo_out = self.model(frame, verbose=False)[0]

        # 2. DeepSORT 트래킹
        tracks = self.tracker.update(yolo_out, frame)

        # 3. depth 융합 + velocity 계산
        results = []
        img_h, img_w = frame.shape[:2]

        for t in tracks:
            depth_m = -1.0
            if depth_map is not None:
                depth_m = extract_depth(
                    depth_map, t["bbox"],
                    (img_h, img_w),
                    depth_map.shape[:2],
                )

            # velocity: 이전 프레임 depth와의 차분
            prev = self.prev_depths.get(t["track_id"], depth_m)
            velocity = (depth_m - prev) if (depth_m > 0 and prev > 0) else 0.0
            self.prev_depths[t["track_id"]] = depth_m

            results.append(TrackedObject(
                track_id = t["track_id"],
                class_id = t["class_id"],
                bbox     = t["bbox"],
                depth_m  = depth_m,
                conf     = t["conf"],
                velocity = velocity,
            ))

        return results