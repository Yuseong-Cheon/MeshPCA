# 실행 환경

## 검증된 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04 userland on WSL2 |
| Kernel | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 |
| Python | 3.10.12 |
| GUI | WSLg / X11-compatible VTK window |
| Mesh unit | meter |

Python 패키지 버전은 `requirements.txt`에 고정돼 있다. `vedo` GUI는 VTK `9.7.0`으로
검증했다.

## 새 환경 설치

Ubuntu/WSL2:

```bash
sudo apt update
sudo apt install -y python3.10-venv libgl1 libglib2.0-0 libxrender1 libxext6
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

PyMeshLab import가 OpenGL library 오류로 실패할 때만 다음을 추가한다.

```bash
sudo apt install -y libopengl0
```

## GUI 환경

WSLg나 일반 Linux desktop에서는 먼저 현재 값을 확인한다.

```bash
echo "$DISPLAY"
echo "$WAYLAND_DISPLAY"
```

정상 desktop session이면 값을 덮어쓰지 않는다. 현재 검증 장비의 fallback은 다음과 같다.

```bash
export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export MPLCONFIGDIR=/tmp/matplotlib-rora
```

`MPLCONFIGDIR`은 홈 설정 폴더가 read-only인 환경에서 Matplotlib 경고와 시작 지연을 막는다.
SSH/headless 환경에서는 self-test는 실행할 수 있지만 HITL GUI 승인은 할 수 없다. X11
forwarding 또는 물리 desktop session이 필요하다.

## 빠른 환경 점검

```bash
.venv/bin/python - <<'PY'
import coacd, matplotlib, numpy, open3d, pymeshlab, scipy, shapely, sklearn, trimesh, vedo, vtk
print("imports: PASS")
print("VTK:", vtk.vtkVersion.GetVTKVersion())
PY
```

## 권장 하드웨어

- RAM 16GB 이상, 고해상도 메시에는 32GB 권장
- GUI를 표시할 OpenGL 3.2 이상 환경
- CoACD와 exact self-intersection 검사는 CPU 시간이 오래 걸릴 수 있음

GPU는 필수 조건이 아니다. 이 패키지는 2DGS 학습이나 RGB-D reconstruction을 포함하지 않는다.

## 흔한 문제

| 증상 | 확인할 것 |
|---|---|
| 창이 안 뜸 | `DISPLAY`, WSLg 실행 여부, VTK import, SSH forwarding |
| 브러시가 안 커짐 | 영문 입력 상태에서 `]`, 작게는 `[` |
| `cut boundary is not a set of closed loops` | open surface 또는 잘못된 face ownership; upstream topology와 paint 경계 확인 |
| `no exact shared interface vertices` | disconnected 경로 결과에 harmonic resize를 사용함; `--joint-anchored-affine` 사용 |
| penetration `>0.5%` | joint origin/axis/range 재검토; 실제 shell overlap일 때만 GUI override 승인 |
| Matplotlib cache 경고 | `MPLCONFIGDIR=/tmp/matplotlib-rora` 설정 |
| 실행 재개 시 hash 오류 | 입력 PLY/config가 바뀜; 새 output 폴더에서 다시 승인 |

