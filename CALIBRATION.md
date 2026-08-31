# RB5–D456 and table calibration

이 보정은 MeshPCA가 구한 물체 위치를 RB5 베이스 좌표로 사용하거나 책상과의 충돌을
검사할 때 필요하다. 물체의 길이·폭·두께만 측정한다면 필수 단계가 아니다.

전체 순서는 다음과 같다.

```text
12개 로봇 자세 촬영
  -> SAM 로봇 마스크
  -> EasyHeC eye-to-hand 최적화
  -> camera <-> robot base 변환
  -> D456 책상 Depth 평면 적합
  -> robot base 기준 책상 높이·기울기·작업영역
```

카메라, 로봇 베이스 또는 카메라 마운트를 움직이면 두 보정을 이 순서로 다시 한다.
책상 기능은 `pyrealsense2`와 Tk GUI가 추가로 필요하다.

## 1. SAM + EasyHeC eye-to-hand calibration

고정된 외부 카메라가 로봇 베이스에 대해 어디에 있는지를 구한다. 각 자세의 실제 로봇
SAM 마스크와 URDF에서 렌더링한 실루엣이 겹치도록 EasyHeC가 외부 파라미터를
최적화한다. SAM은 로봇을 검출하는 도구이고, 최종 보정값은 EasyHeC가 계산한다.

검증한 장비 구성은 RB5-850E, AIDIN AFT200, 완전히 열린 Robotiq 2F-85, Intel
RealSense D456이다. RB5용 캡처 도구가 추가된 [EasyHeC](https://github.com/ootts/EasyHeC)
checkout 루트에서 다음 순서로 실행한다.

```bash
conda activate easyhec-rb5
python tools/rb5_xhand/prepare_robotiq_assets.py
tools/rb5_xhand/capture_ui/run_robotiq.sh
```

웹 UI에서 로봇이 정지한 상태로 서로 다른 자세 12장을 촬영한다. 손목 회전과 팔꿈치
높이를 다양하게 하고 로봇 전체가 영상 안에 들어오게 한다. 같은 자세를 반복 촬영해도
보정 정보는 늘지 않는다. 촬영 중에는 카메라와 그리퍼 형상을 바꾸지 않는다.

```bash
export EASYHEC_ASSET_DIR="$PWD/assets/rb5_robotiq"
NAME=cap_YYYYMMDD_HHMM
D=data/rb5_robotiq_captures/$NAME

python tools/rb5_xhand/import_data.py \
  --name "$NAME" --images "$D/color" --qpos "$D/qpos" --K "$D/K.txt"
python tools/rb5_xhand/annotate_masks.py --name "$NAME" --screen-scale 2.0
python tools/rb5_xhand/init_pose.py -c "configs/rb5_xhand/$NAME.yaml" --search
python tools/rb5_xhand/solve.py -c "configs/rb5_xhand/$NAME.yaml"
python tools/rb5_xhand/render_check.py -c "configs/rb5_xhand/$NAME.yaml"
python tools/rb5_xhand/render_robot.py -c "configs/rb5_xhand/$NAME.yaml"
```

SAM UI에서는 팔, 손목 센서, 그리퍼까지 실제 렌더 자산에 포함된 전체 형상을 선택하고
사람 손, 케이블, 배경은 제외한다. 초기 실루엣 탐색의 mean IoU가 약 0.6 이상인지 먼저
확인하고, 최종 결과는 숫자뿐 아니라 모든 프레임의 오버레이도 확인한다.

최종 파일은 `models/rb5_xhand/<NAME>/Tc_c2b.txt`이다. 좌표 규약은 다음과 같다.

```text
p_camera = Tc_c2b @ p_base
base_from_camera = inverse(Tc_c2b)
```

- `base`: RB5 URDF의 `left_link0`, 단위 m, +z 위쪽
- `camera`: D456 color optical frame, OpenCV 축(x 오른쪽, y 아래, z 전방)
- 카메라가 움직이지 않은 경우에만 이전 결과를 다음 세션 초기값으로 재사용한다.

EasyHeC 결과를 이 저장소의 공통 JSON 형식으로 변환한다.

```bash
python calibration/import_easyhec.py \
  --transform "models/rb5_xhand/$NAME/Tc_c2b.txt" \
  --intrinsics "data/rb5_xhand/$NAME/K.txt" \
  --output calibration/handeye.json
```

## 2. D456 table-plane calibration

이 단계는 Depth 센서 자체를 보정하는 작업이 아니다. D456의 aligned metric Depth를 위의
`base_from_camera`로 RB5 베이스 좌표에 옮긴 뒤 책상 평면을 찾는 작업이다.

```bash
python calibration/table_rgbd.py \
  --handeye calibration/handeye.json \
  --output calibration/table.json
```

1. D456 color/depth를 정렬하고 여러 depth 프레임의 중앙값을 취한다.
2. UI에서 책상만 포함하는 다각형을 왼쪽 클릭으로 지정한다.
3. 각 depth 픽셀을 카메라 3D 점으로 역투영한다.
4. `base_from_camera`를 적용해 로봇 베이스 좌표로 변환한다.
5. RANSAC으로 이상점을 제거하고 SVD로 평면을 다시 적합한다.

UI 조작은 `왼쪽 클릭: 경계점`, `Enter: 적합`, `Backspace: 마지막 점 취소`,
`R: 초기화`, `Esc: 취소`이다. 결과에는 아래 항목을 저장한다.

```json
{
  "status": "valid",
  "plane_in_robot_base": {
    "equation": ["nx", "ny", "nz", "d"],
    "height_at_base_origin_m": "-d / nz",
    "tilt_deg": "angle from base +z"
  },
  "quality": {
    "inlier_fraction": "RANSAC inlier ratio",
    "rms_mm": "inlier plane residual"
  }
}
```

현재 품질 기준은 inlier fraction 0.75 이상, RMS 4 mm 이하, base +z 대비 기울기
5도 이하이다. 선택 영역에는 책상 위 물체, 로봇, 바닥을 넣지 않는다.

생성되는 hand-eye JSON, table JSON, overlay PNG와 촬영 세션은 장비별 결과이므로 Git에
추가하지 않는다.
