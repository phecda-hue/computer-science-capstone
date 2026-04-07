# 파이프라인 전체 명세
버전: v1.0 | 작성일: 2025-04-03

## 시스템 개요
핸드폰 카메라로 촬영한 영상을 Python 서버에서 처리하고,
결과 데이터를 앱으로 넘겨 결정 엔진과 알림을 구동하는 구조.

## 담당 분리
| 단계              | 담당       | 구현 언어  |
|-------------------|------------|------------|
| 카메라 캡처        | 유선우         | Kotlin     |
| 전처리             | 유선우         | Kotlin     |
| YOLO11 추론        | 오준석, 오학균 | Python     |
| DeepSORT 트래킹    | 오준석, 오학균 | Python     |
| Depth Anything v2 | 오준석, 오학균 | Python     |
| 데이터 융합        | 오준석, 오학균 | Python     |
| 앱으로 전달        | 공통          | REST / Socket |
| 결정 엔진          | 유선우, 오준석 | Kotlin     |
| 알림 출력          | 유선우        | Kotlin     |

## 처리 흐름
카메라 프레임
  → [전처리] 리사이즈(640×640) · BGR→RGB · float32 정규화
  → [YOLO11 + DeepSORT] bbox · class_id · track_id · conf
  → [Depth Anything v2] 픽셀별 깊이 (미터, float32)
  → [데이터 융합] TrackedObject 리스트 생성
  → [API 전달] JSON over HTTP / WebSocket
  → [결정 엔진] 위험도 판단
  → [알림] 음성·진동·화면