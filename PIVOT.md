# PIVOT real-robot stages

이 디렉터리는 PIVOT 실물 파이프라인을 단계별 실행 코드로 보관한다. 센서 측정값,
캘리브레이션 JSON, 이미지와 실험 결과는 Git에 넣지 않는다.

```text
1. EasyHeC/SAM hand-eye       완료: calibration/import_easyhec.py
2. D456 책상 평면            완료: calibration/table_rgbd.py
3. RB5 + AFT 연결/3자세 타어 완료: pivot/
4. Robotiq 열기/닫기          다음 단계
5. 물체 자세·렌치 측정        예정
6. PIVOT 밀도 추정/URDF       예정
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
