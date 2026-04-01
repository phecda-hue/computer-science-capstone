import numpy as np

def extract_depth(depth_map: np.ndarray,
                  bbox: list,
                  img_shape: tuple,
                  depth_shape: tuple) -> float:
    """
    depth_map : ARCore에서 받은 깊이 맵 (단위: mm)
    bbox      : [x1, y1, x2, y2] (이미지 해상도 기준)
    img_shape : (H, W) 원본 이미지 해상도
    depth_shape: (H, W) depth map 해상도 (ARCore는 다를 수 있음)
    반환      : 미터 단위 깊이값 (감지 실패 시 -1.0)
    """
    # bbox를 depth map 해상도로 리스케일
    sx = depth_shape[1] / img_shape[1]
    sy = depth_shape[0] / img_shape[0]

    x1, y1, x2, y2 = bbox
    dx1 = max(0, int(x1 * sx))
    dy1 = max(0, int(y1 * sy))
    dx2 = min(depth_shape[1], int(x2 * sx))
    dy2 = min(depth_shape[0], int(y2 * sy))

    roi = depth_map[dy1:dy2, dx1:dx2]

    # bbox 중심 20% 영역만 사용 (경계 노이즈 제거)
    h, w = roi.shape
    cy, cx = h // 2, w // 2
    margin_y, margin_x = max(1, h // 5), max(1, w // 5)
    center_roi = roi[cy-margin_y:cy+margin_y, cx-margin_x:cx+margin_x]

    # 유효값 필터 (0=무효, 8000mm=8m 이상 노이즈)
    valid = center_roi[(center_roi > 0) & (center_roi < 8000)]

    if len(valid) == 0:
        return -1.0

    return float(np.median(valid)) / 1000.0  # mm → m