#!/usr/bin/env python3
"""PIVOT용 AFT200 모니터와 Robotiq 2F-85 통합 UI."""

import argparse
import glob
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import font, messagebox, ttk

try:
    from .aft_tare import Aft200Sensor, TareTable
    from .robotiq import Robotiq2F85
except ImportError:  # 직접 `python pivot/rb5_ui.py`로 실행할 때
    from aft_tare import Aft200Sensor, TareTable
    from robotiq import Robotiq2F85


AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
AXIS_LABELS = ("Fx (N)", "Fy (N)", "Fz (N)",
               "Tx (N·m)", "Ty (N·m)", "Tz (N·m)")


def prepare_korean_font():
    """X11 core-font Tk에 설치된 나눔고딕을 노출한다."""
    source_dir = Path("/usr/share/fonts/truetype/nanum")
    tools = (shutil.which("mkfontscale"), shutil.which("mkfontdir"), shutil.which("xset"))
    sources = [source_dir / "NanumGothic.ttf", source_dir / "NanumGothicBold.ttf"]
    if not all(tools) or not all(path.is_file() for path in sources):
        return "nimbus sans l"
    cache = Path(tempfile.gettempdir()) / f"pivot-xfonts-{os.getuid()}"
    cache.mkdir(exist_ok=True)
    for source in sources:
        target = cache / source.name
        if not os.path.lexists(target):
            target.symlink_to(source)
    # ponytail: 현재 X11/Tk의 fontconfig 미지원 우회. Xft 지원 Tk면 이 블록을 지운다.
    subprocess.run([tools[0], str(cache)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([tools[1], str(cache)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([tools[2], "+fp", str(cache)], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([tools[2], "fp", "rehash"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "nanumgothic"


class DisplayFilter:
    """원값을 읽기 쉽게 부드럽게 만드는 화면용 EMA."""

    def __init__(self, alpha=0.1):
        self.alpha = float(alpha)
        self.state = None

    def update(self, values):
        values = tuple(float(value) for value in values)
        self.state = (values if self.state is None else
                      tuple(old + self.alpha * (new - old)
                            for old, new in zip(self.state, values)))
        return self.state


class App(tk.Tk):
    def __init__(self, host, port, tare_path, ui_font):
        super().__init__()
        self.title("PIVOT · RB5 AFT200 + Robotiq 2F-85")
        self.geometry("980x760")
        self.minsize(900, 700)
        self.protocol("WM_DELETE_WINDOW", self.shutdown)
        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont"):
            font.nametofont(name).configure(family=ui_font, size=12)
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TButton", font=(ui_font, 11), padding=(10, 7))
        style.configure("TLabelframe.Label", font=(ui_font, 13, "bold"))
        style.configure("Title.TLabel", font=(ui_font, 18, "bold"))
        style.configure("Header.TLabel", font=(ui_font, 12, "bold"))
        style.configure("Value.TLabel", font=(ui_font, 13))
        style.configure("Filtered.TLabel", font=(ui_font, 13),
                        foreground="#075985")
        style.configure("Mass.TLabel", font=(ui_font, 15, "bold"),
                        foreground="#1d4ed8")
        style.configure("Help.TLabel", font=(ui_font, 11),
                        foreground="#475569")
        style.configure("Danger.TLabel", font=(ui_font, 11, "bold"),
                        foreground="#b91c1c")

        self.host = host
        self.tare_path = tare_path
        self.samples = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.display_filter = DisplayFilter()
        self.gripper = Robotiq2F85(port)
        self.gripper_keepalive_started = False
        self.sensor_status = tk.StringVar(value="AFT 연결: 확인 중")
        self.pivot_tare_status = tk.StringVar()
        self.gripper_status = tk.StringVar(value="연결 안 됨")
        self.raw_vars = [tk.StringVar(value="-") for _ in AXES]
        self.filtered_vars = [tk.StringVar(value="-") for _ in AXES]
        self.port = tk.StringVar(value=port)
        self.speed = tk.IntVar(value=64)
        self.force = tk.IntVar(value=32)
        self._build()
        self.refresh_ports()
        self.refresh_pivot_tare()
        threading.Thread(target=self.sensor_worker, daemon=True).start()
        if self.port.get() in self.port_box["values"]:
            self.after(100, self.connect_gripper)
        self.after(20, self.consume_samples)

    def _build(self):
        ttk.Label(self, text="PIVOT 하드웨어 상태", style="Title.TLabel").pack(
            anchor="w", padx=18, pady=(14, 4))
        sensor = ttk.LabelFrame(self, text="AIDIN AFT200 · RB5 Modbus TCP")
        sensor.pack(fill="x", padx=18, pady=10)
        for column in range(4):
            sensor.columnconfigure(column, weight=1)
        ttk.Label(sensor, text=f"제어박스 IP  {self.host}", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 4))
        ttk.Label(sensor, textvariable=self.sensor_status, style="Header.TLabel").grid(
            row=0, column=2, columnspan=2, sticky="e", padx=12, pady=(10, 4))
        ttk.Label(sensor, style="Help.TLabel", text=(
            "센서 원값은 방금 읽은 값(노이즈 포함), 필터값은 원값을 부드럽게 한 "
            "화면 표시용 값입니다. 둘 다 그리퍼 무게가 포함된 절대 센서값입니다."),
            wraplength=880, justify="left").grid(
                row=1, column=0, columnspan=4, sticky="w", padx=12, pady=(2, 10))

        ttk.Label(sensor, text="축", style="Header.TLabel").grid(row=2, column=0)
        ttk.Label(sensor, text="센서 원값 (즉시)", style="Header.TLabel").grid(
            row=2, column=1)
        ttk.Label(sensor, text="필터값 (부드럽게)", style="Header.TLabel").grid(
            row=2, column=2)
        for row, (name, raw, filtered) in enumerate(
                zip(AXIS_LABELS, self.raw_vars, self.filtered_vars), 3):
            ttk.Label(sensor, text=name, style="Header.TLabel").grid(
                row=row, column=0, sticky="w", padx=12, pady=2)
            ttk.Label(sensor, textvariable=raw, width=18, style="Value.TLabel").grid(
                row=row, column=1, pady=2)
            ttk.Label(sensor, textvariable=filtered, width=18,
                      style="Filtered.TLabel").grid(row=row, column=2, pady=2)
        ttk.Label(sensor, textvariable=self.pivot_tare_status, style="Header.TLabel").grid(
            row=9, column=0, columnspan=3, sticky="w", padx=12, pady=(12, 10))
        ttk.Button(sensor, text="PIVOT 타어 다시 읽기", command=self.refresh_pivot_tare).grid(
            row=9, column=3, sticky="e", padx=12, pady=(12, 10))

        grip = ttk.LabelFrame(self, text="Robotiq 2F-85 · USB/RS485")
        grip.pack(fill="x", padx=18, pady=10)
        grip.columnconfigure(1, weight=1)
        ttk.Label(grip, text="USB 포트", style="Header.TLabel").grid(
            row=0, column=0, padx=12, pady=10)
        self.port_box = ttk.Combobox(grip, textvariable=self.port, width=18)
        self.port_box.grid(row=0, column=1, sticky="ew", padx=6, pady=10)
        ttk.Button(grip, text="새로고침", command=self.refresh_ports).grid(
            row=0, column=2, padx=6)
        ttk.Button(grip, text="다시 연결", command=self.connect_gripper).grid(
            row=0, column=3, padx=6)
        ttk.Label(grip, textvariable=self.gripper_status, style="Header.TLabel").grid(
            row=0, column=4, padx=12)

        ttk.Label(grip, text="속도", style="Header.TLabel").grid(row=1, column=0)
        ttk.Scale(grip, from_=1, to=255, variable=self.speed, orient="horizontal").grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Label(grip, textvariable=self.speed, width=4).grid(row=1, column=4)
        ttk.Label(grip, text="힘", style="Header.TLabel").grid(row=2, column=0)
        ttk.Scale(grip, from_=1, to=255, variable=self.force, orient="horizontal").grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Label(grip, textvariable=self.force, width=4).grid(row=2, column=4)
        ttk.Button(grip, text="Reset + Activate", command=self.initialize_gripper).grid(
            row=3, column=0, columnspan=2, padx=12, pady=12)
        ttk.Button(grip, text="열기", command=lambda: self.move_gripper(False)).grid(
            row=3, column=2, padx=8)
        ttk.Button(grip, text="닫기", command=lambda: self.move_gripper(True)).grid(
            row=3, column=3, padx=8)

        ttk.Label(self, style="Danger.TLabel", text=(
            "센서 원값을 0으로 만들지 않습니다. PIVOT 측정 단계가 저장된 "
            "3방향 6축 타어를 자동으로 빼서 물체 렌치를 계산합니다."), wraplength=920).pack(
                anchor="w", padx=22, pady=(8, 2))
        ttk.Label(self, text="그리퍼 동작 전 손과 물체를 손가락 사이에서 치우세요.",
                  style="Danger.TLabel").pack(anchor="w", padx=22, pady=(2, 10))

    def sensor_worker(self):
        while not self.stop_event.is_set():
            try:
                sensor = Aft200Sensor(self.host, hz=50.0)
                self.samples.put(("status", "AFT 연결: 정상"))
                for values in sensor.stream():
                    if self.stop_event.is_set():
                        return
                    self.samples.put(("sample", values))
            except Exception as error:
                self.samples.put(("status", f"AFT 연결 오류: {error}"))
                self.stop_event.wait(1.0)

    def consume_samples(self):
        try:
            while True:
                kind, value = self.samples.get_nowait()
                if kind == "status":
                    self.sensor_status.set(value)
                    continue
                filtered = self.display_filter.update(value)
                for variable, number in zip(self.raw_vars, value):
                    variable.set(f"{number:.3f}")
                for variable, number in zip(self.filtered_vars, filtered):
                    variable.set(f"{number:.3f}")
        except queue.Empty:
            pass
        if not self.stop_event.is_set():
            self.after(20, self.consume_samples)

    def refresh_pivot_tare(self):
        try:
            table = TareTable.load(self.tare_path)
            count = len(table.table)
            label = "준비됨" if count == 3 else "불완전"
            self.pivot_tare_status.set(
                f"밀도 추정용 PIVOT 6축 타어: {label} ({count}/3 방향)")
        except Exception as error:
            self.pivot_tare_status.set(f"PIVOT 6축 타어 없음/오류: {error}")

    def refresh_ports(self):
        ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        self.port_box["values"] = ports
        if ports and self.port.get() not in ports:
            self.port.set(ports[0])

    @staticmethod
    def status_text(status):
        return (f"활성={int(status['activated'])} · fault=0x{status['fault']:02X} · "
                f"위치={status['position']}")

    def run_gripper(self, action):
        def worker():
            try:
                status = action()
                self.after(0, self.gripper_status.set, self.status_text(status))
            except Exception as error:
                self.after(0, self.gripper_status.set, "통신 실패")
                self.after(0, messagebox.showerror, "Robotiq 오류", str(error))
        threading.Thread(target=worker, daemon=True).start()

    def connect_gripper(self):
        self.gripper.port = self.port.get()
        def connect():
            status = self.gripper.connect()
            if not self.gripper_keepalive_started:
                self.gripper_keepalive_started = True
                threading.Thread(target=self.gripper_keepalive, daemon=True).start()
            return status
        self.run_gripper(connect)

    def gripper_keepalive(self):
        while not self.stop_event.wait(0.5):
            if not self.gripper.connected:
                continue
            try:
                status = self.gripper.status()
                self.after(0, self.gripper_status.set, self.status_text(status))
            except Exception as error:
                self.after(0, self.gripper_status.set, f"통신 오류: {error}")

    def initialize_gripper(self):
        self.run_gripper(self.gripper.initialize)

    def move_gripper(self, close):
        speed, force = int(self.speed.get()), int(self.force.get())
        action = self.gripper.close_gripper if close else self.gripper.open
        self.run_gripper(lambda: action(speed, force))

    def shutdown(self):
        self.stop_event.set()
        self.gripper.close()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.50.51")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--tare", default="calibration/aft_tare_current.json")
    args = parser.parse_args()
    App(args.host, args.port, args.tare, prepare_korean_font()).mainloop()


if __name__ == "__main__":
    main()
