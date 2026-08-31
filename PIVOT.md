# PIVOT real-robot stages

이 디렉터리는 PIVOT 실물 파이프라인을 단계별 실행 코드로 보관한다. 센서 측정값,
캘리브레이션 JSON, 이미지와 실험 결과는 Git에 넣지 않는다.

```text
1. EasyHeC/SAM hand-eye       완료: calibration/import_easyhec.py
2. D456 책상 평면            완료: calibration/table_rgbd.py
3. RB5 + AFT 연결/3자세 타어 완료: pivot/
4. Robotiq 열기/닫기          완료: pivot/robotiq.py
5. AFT + Robotiq 통합 UI      완료: pivot/rb5_ui.py
6. 물체 자세·렌치 측정        다음 단계
7. PIVOT 밀도 추정/URDF       예정
```

## 3. AFT200 3자세 타어

타어는 물체를 잡지 않은 상태에서 그리퍼와 마운트가 만드는 6축 렌치를 중력 방향별로
기록한다. 타어 때와 물체 측정 때 Robotiq 개구량이 같아야 한다.

`pivot/tare_real.py`는 PIVOT의 기존 Drake 장면, IK, RRT 충돌 계획기와 RB5 드라이버를
재사용한다. PIVOT 체크아웃의 `my_work`에서 실행한다.

```bash
cd ~/Desktop/PIVOT/my_work

# 이동 명령 없이 연결과 계획만 확인
PIVOT_WORKDIR=$PWD \
  ../robot_learning/scripts/run_drake_env.sh python \
  ~/MeshPCA/pivot/tare_real.py --plan-only

# 빈 그리퍼와 깨끗한 작업영역을 확인한 뒤 실제 실행
PIVOT_WORKDIR=$PWD \
  ../robot_learning/scripts/run_drake_env.sh python \
  ~/MeshPCA/pivot/tare_real.py \
  --output calibration/aft_tare_current.json --overwrite
```

실행 순서는 `g-down -> g-x -> g-y -> 시작 자세 복귀`이다. 각 경로는 먼저 충돌 검사를
통과해야 하며, 실제 이동 직전에 로봇 자세가 바뀌면 중단한다. 각 자세에서는 두 번의
평균값 차이가 힘 0.5 N, 토크 0.05 N·m 이하일 때만 기록한다.

코드 자체 검사는 로봇을 연결하거나 움직이지 않는다.

```bash
python -m unittest pivot/test_aft_tare.py
```

생성되는 `pivot/calibration/*.json`, `pivot/results/`와 기존 `calibration/*.json`은
`.gitignore`로 제외된다.

## 4. Robotiq 2F-85

Linux `/dev/ttyUSB0`, 115200 8N1, Modbus RTU slave 9 연결을 사용한다. 별도 serial
패키지는 필요 없다.

```bash
python pivot/robotiq.py status
python pivot/robotiq.py initialize
python pivot/robotiq.py close --speed 64 --force 32
python pivot/robotiq.py open --speed 64 --force 32
```

`0x09`는 1초 이상 통신이 없을 때 생기는 경고라서 이동 중에는 완료될 때까지 상태를
폴링한다. 실제 동작 전 손가락 사이를 비운다.

## 5. 원본 통합 UI의 PIVOT 연결

```bash
python pivot/rb5_ui.py \
  --host 192.168.50.51 \
  --port /dev/ttyUSB0 \
  --tare calibration/aft_tare_current.json
```

UI에는 AFT200의 실제 6축 원값과 이를 부드럽게 한 EMA 필터값만 표시한다. Fz를 g로
환산한 원본 UI의 간이 질량과 화면 영점은 PIVOT에 쓰이지 않아 제거했다. UI의 타어
표시는 파일의 3방향 데이터가 준비됐는지만 확인하며 화면 숫자에서는 빼지 않는다.

실제 측정 단계에서는 같은 중력 방향의 빈 그리퍼 6축 값을 센서 6축 값에서 뺀다.

```python
tare = TareTable.load("calibration/aft_tare_current.json")
object_wrench = tare.apply(g_hat, sensor_wrench)
```

UI는 AFT와 USB 포트를 자동 연결하고 Robotiq에 0.5초 상태 heartbeat를 보낸다. X11의
Tk가 시스템 한글 폰트를 찾지 못하는 환경에서는 설치된 NanumGothic을 자동 등록한다.
