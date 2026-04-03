import numpy as np

# IIR 필터 상태 저장 (track_id → 필터링된 depth)
_iir_state: dict[int, float] = {}
IIR_ALPHA = 0.3  # 낮을수록 필터 강함 (떨림 제거), 높을수록 반응 빠름

# ROI 설정: 지면으로부터 일정 높이 이내 (이미지 하단 비율)
ROI_BOTTOM_RATIO = 1.0   # 이미지 하단 100%
ROI_TOP_RATIO    = 0.5   # 이미지 높이의 50% 지점부터 (상단 절반 무시)

# 노이즈 필터 설정
DEPTH_JUMP_THRESHOLD = 0.5   # 주변 픽셀과 거리차 이 값 초과 시 무시 (미터)
MIN_VALID_PIXELS     = 10    # 이 개수 미만이면 장애물로 간주하지 않음
DEPTH_MIN_M          = 0.1   # 유효 최소 거리 (미터)
DEPTH_MAX_M          = 8.0   # 유효 최대 거리 (미터)


def apply_roi(depth_map: np.ndarray) -> np.ndarray:
    """
    지면 기준 일정 높이 이내 ROI만 남기고 나머지는 0으로 마스킹
    """
    h = depth_map.shape[0]
    roi = depth_map.copy()
    top_px = int(h * ROI_TOP_RATIO)
    roi[:top_px, :] = 0  # 상단 절반 무시
    return roi


def filter_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    주변 픽셀과 거리차가 너무 큰 점 제거 (중앙값 기반)
    """
    from scipy.ndimage import median_filter

    # 3×3 중앙값 필터로 주변 픽셀 기준값 계산
    median = median_filter(depth_map.astype(np.float32), size=3)

    # 중앙값과 차이가 threshold 초과인 픽셀 무효화
    mask = np.abs(depth_map.astype(np.float32) - median) > DEPTH_JUMP_THRESHOLD
    filtered = depth_map.copy().astype(np.float32)
    filtered[mask] = 0
    return filtered


def extract_depth(depth_map: np.ndarray,
                  bbox: list,
                  img_shape: tuple,
                  depth_shape: tuple,
                  track_id: int) -> float:
    """
    bbox 영역에서 ROI · 노이즈 필터 · IIR 필터를 거쳐 깊이 추출

    반환: IIR 필터링된 깊이 (미터), 실패 시 -1.0
    """
    # 1. ROI 적용
    roi_map = apply_roi(depth_map)

    # 2. 주변 픽셀 노이즈 제거
    clean_map = filter_depth_map(roi_map)

    # 3. bbox를 depth_map 해상도로 리스케일
    sx = depth_shape[1] / img_shape[1]
    sy = depth_shape[0] / img_shape[0]
    x1, y1, x2, y2 = bbox
    dx1 = max(0, int(x1 * sx))
    dy1 = max(0, int(y1 * sy))
    dx2 = min(depth_shape[1], int(x2 * sx))
    dy2 = min(depth_shape[0], int(y2 * sy))

    crop = clean_map[dy1:dy2, dx1:dx2]

    # 4. 유효 픽셀 필터링
    valid = crop[(crop > DEPTH_MIN_M) & (crop < DEPTH_MAX_M)]

    # 5. 최소 픽셀 수 미만이면 장애물로 간주하지 않음
    if len(valid) < MIN_VALID_PIXELS:
        return -1.0

    raw_depth = float(np.median(valid))

    # 6. IIR 필터 적용 (휠체어 떨림 노이즈 제거)
    #    y_n = alpha * x_n + (1 - alpha) * y_{n-1}
    prev = _iir_state.get(track_id, raw_depth)
    filtered_depth = IIR_ALPHA * raw_depth + (1 - IIR_ALPHA) * prev
    _iir_state[track_id] = filtered_depth

    return filtered_depth


def clear_iir_state(track_id: int):
    """트랙 소실 시 IIR 상태 제거"""
    _iir_state.pop(track_id, None)