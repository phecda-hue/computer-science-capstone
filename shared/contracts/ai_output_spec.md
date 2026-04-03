# AI 파이프라인 출력 명세
버전: v1.0

## 전달 방식
- 프로토콜: HTTP POST 또는 WebSocket (데모 환경에 맞게 협의)
- 포맷: JSON
- 주기: 매 프레임 (목표 15fps)

## TrackedObject 구조
```json
{
  "frame_id": 142,
  "timestamp": 1712134800.123,
  "objects": [
    {
      "track_id":  3,
      "class_id":  5,
      "class_name": "pedestrian",
      "bbox":      [120, 80, 300, 420],
      "conf":      0.91,
      "depth_m":   2.34,
      "velocity":  -0.18
    }
  ]
}
```

## 필드 설명
| 필드        | 타입        | 설명                                      |
|-------------|-------------|-------------------------------------------|
| frame_id    | int         | 프레임 순번                               |
| timestamp   | float       | Unix 타임스탬프 (초)                      |
| track_id    | int         | DeepSORT 트랙 ID (프레임 간 유지)         |
| class_id    | int         | class_ids.yaml 기준 (0~11)                |
| class_name  | string      | 클래스 이름                               |
| bbox        | [x1,y1,x2,y2] | 픽셀 좌표, 원본 해상도 기준             |
| conf        | float       | YOLO 신뢰도 (0.0~1.0)                    |
| depth_m     | float       | 미터 단위 깊이, -1.0이면 추정 실패        |
| velocity    | float       | depth 변화량 (m/frame), 음수=접근 중      |

## 예외 처리
- depth_m = -1.0 → 깊이 추정 실패, 결정 엔진에서 depth 무시
- objects = [] → 감지된 객체 없음
- track_id는 객체 소실 후 재등장 시 새 ID 부여됨