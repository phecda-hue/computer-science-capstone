import numpy as np
from ultralytics import YOLO
from engine.tracker_bridge import TrackerBridge
from engine.depth_fusion import extract_depth, clear_iir_state
from engine.risk_calculator import calc_risk, RiskResult
from dataclasses import dataclass, field

@dataclass
class TrackedObject:
    track_id:   int
    class_id:   int
    bbox:       list
    depth_m:    float
    conf:       float
    velocity:   float
    risk:       RiskResult | None = None


class Pipeline:
    def __init__(self, model_path: str):
        self.model      = YOLO(model_path)
        self.tracker    = TrackerBridge()
        self.prev_depths: dict[int, float] = {}
        self.prev_time:   float = 0.0

    def process(self,
                frame: np.ndarray,
                depth_map: np.ndarray | None = None,
                timestamp: float = 0.0) -> list[TrackedObject]:

        # 1. YOLO 탐지
        yolo_out = self.model(frame, verbose=False)[0]

        # 2. DeepSORT 트래킹
        tracks = self.tracker.update(yolo_out, frame)

        img_h, img_w = frame.shape[:2]
        dt = max(timestamp - self.prev_time, 1/30)  # 프레임 간격 (초)
        self.prev_time = timestamp

        active_ids = set()
        results = []

        for t in tracks:
            tid = t["track_id"]
            active_ids.add(tid)

            # 3. depth 추출 (IIR 필터 포함)
            depth_m = -1.0
            if depth_map is not None:
                depth_m = extract_depth(
                    depth_map, t["bbox"],
                    (img_h, img_w),
                    depth_map.shape[:2],
                    tid,
                )

            # 4. velocity 계산 (m/s)
            #    v = (d_현재 - d_이전) / dt
            #    음수 = 접근 중
            prev_d = self.prev_depths.get(tid, depth_m)
            if depth_m > 0 and prev_d > 0:
                velocity = (depth_m - prev_d) / dt
            else:
                velocity = 0.0
            self.prev_depths[tid] = depth_m

            # 5. bbox 면적 (depth fallback용)
            x1, y1, x2, y2 = t["bbox"]
            bbox_area = (x2 - x1) * (y2 - y1)

            # 6. 위험도 계산
            risk = calc_risk(
                track_id  = tid,
                class_id  = t["class_id"],
                depth_m   = depth_m,
                velocity  = velocity,
                bbox_area = bbox_area,
            )

            results.append(TrackedObject(
                track_id = tid,
                class_id = t["class_id"],
                bbox     = t["bbox"],
                depth_m  = depth_m,
                conf     = t["conf"],
                velocity = velocity,
                risk     = risk,
            ))

        # 소실된 트랙 IIR 상태 정리
        lost_ids = set(self.prev_depths.keys()) - active_ids
        for lid in lost_ids:
            clear_iir_state(lid)
            del self.prev_depths[lid]

        # 위험도 높은 순으로 정렬
        results.sort(key=lambda o: o.risk.risk_score if o.risk else 0,
                     reverse=True)
        return results