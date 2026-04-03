# engine/tracker_bridge.py
from deep_sort_realtime.deepsort_tracker import DeepSort

class TrackerBridge:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,        # 30프레임 안 보이면 트랙 삭제
            n_init=3,          # 3프레임 연속 감지돼야 트랙 확정
            max_iou_distance=0.7,
        )

    def update(self, yolo_results, frame):
        """
        yolo_results: YOLO model(frame)[0].boxes 결과
        frame: BGR numpy array
        반환: [track_id, x1, y1, x2, y2, class_id, conf] 리스트
        """
        detections = []
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            # DeepSORT 입력 포맷: ([x1,y1,w,h], conf, class_id)
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, cls))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()          # [x1, y1, x2, y2]
            cls  = track.get_det_class()
            conf = track.get_det_conf() or 0.0
            results.append({
                "track_id": tid,
                "bbox":     ltrb,
                "class_id": cls,
                "conf":     conf,
            })

        return results