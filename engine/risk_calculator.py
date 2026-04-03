from dataclasses import dataclass

# 클래스별 가중치 W (질량 대체)
CLASS_WEIGHTS: dict[int, float] = {
    0:  0.8,   # stairs
    1:  0.6,   # curb
    2:  0.5,   # manhole
#    3:  0.5,   # step
    3:  0.5,   # person
#    5:  0.7,   # bicycle
#    6:  1.0,   # vehicle
#    7:  0.4,   # pole
#    8:  0.6,   # construction
#    9: 0.4,   # protrusion
    4: 0.9,   # hole          ← 낙상 위험 높음
}

# 위험도 임계값
RISK_HIGH   = 2.0
RISK_MEDIUM = 0.8
RISK_LOW    = 0.3

# bbox 면적 기반 속도 추정용 (depth 없을 때 fallback)
BBOX_AREA_SCALE = 1e-5


@dataclass
class RiskResult:
    track_id:   int
    class_id:   int
    class_name: str
    depth_m:    float
    v_approach: float   # 접근 속도 (m/s), 0 이상만 (멀어지면 0)
    weight:     float   # 클래스 가중치 W
    risk_score: float   # Risk = W * (1/d^2 + v_approach/d)
    level:      str     # "HIGH" / "MEDIUM" / "LOW" / "SAFE"


CLASS_NAMES = [
    "stairs", "curb", "manhole", # "step",
    "person", # "bicycle", "vehicle", "pole",
    # "construction", "protrusion", 
    "hole"
]


def calc_risk(track_id: int,
              class_id: int,
              depth_m: float,
              velocity: float,
              bbox_area: float | None = None) -> RiskResult:
    """
    단일 객체의 위험도 계산

    Parameters
    ----------
    track_id   : DeepSORT 트랙 ID
    class_id   : YOLO 클래스 ID
    depth_m    : IIR 필터링된 깊이 (미터), -1이면 추정 실패
    velocity   : depth 변화량 (m/frame), 음수 = 접근 중
    bbox_area  : depth 없을 때 fallback용 bbox 픽셀 면적
    """
    W = CLASS_WEIGHTS.get(class_id, 0.5)
    name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"

    # depth 추정 실패 시 bbox 면적으로 fallback
    if depth_m <= 0:
        if bbox_area and bbox_area > 0:
            # bbox 면적이 클수록 가깝다고 추정
            depth_m = 1.0 / (bbox_area * BBOX_AREA_SCALE + 1e-6)
            depth_m = float(np.clip(depth_m, DEPTH_MIN_M, DEPTH_MAX_M))
        else:
            # 추정 불가 → 위험도 0
            return RiskResult(track_id, class_id, name,
                              -1.0, 0.0, W, 0.0, "SAFE")

    # 접근 속도: 멀어지는 경우(velocity > 0)는 0으로 처리
    # velocity는 depth 변화량이므로 음수 = 접근
    v_relative = -velocity          # 부호 반전: 양수 = 접근
    v_approach = max(0.0, v_relative)

    # Risk = W * (1/d^2 + v_approach/d)
    d = max(depth_m, 0.01)         # 0 나눔 방지
    risk_score = W * (1.0 / d**2 + v_approach / d)

    # 위험 등급 분류
    if risk_score >= RISK_HIGH:
        level = "HIGH"
    elif risk_score >= RISK_MEDIUM:
        level = "MEDIUM"
    elif risk_score >= RISK_LOW:
        level = "LOW"
    else:
        level = "SAFE"

    return RiskResult(
        track_id   = track_id,
        class_id   = class_id,
        class_name = name,
        depth_m    = depth_m,
        v_approach = v_approach,
        weight     = W,
        risk_score = risk_score,
        level      = level,
    )