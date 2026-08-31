#!/usr/bin/env python3
"""PIVOT용 AFT200 모니터와 Robotiq 2F-85 통합 UI."""

import argparse
from collections import deque
import glob
import queue
import statistics
import threading
import tkinter as tk
from tkinter import font, messagebox, ttk

from aft_tare import Aft200Sensor, TareTable
from robotiq import Robotiq2F85


AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
AXIS_LABELS = ("Fx (N)", "Fy (N)", "Fz (N)",
               "Tx (N·m)", "Ty (N·m)", "Tz (N·m)")
GRAVITY = 9.80665


class DisplayFilter:
    """화면 간이 질량에만 쓰는 EMA와 동일 자세 Fz 영점."""

    def __init__(self, alpha=0.1, sample_rate=50, stable_seconds=1, threshold_g=30):
        self.alpha = float(alpha)
        self.state = None
        self.zero_fz = 0.0
        self.window = deque(maxlen=round(sample_rate * stable_seconds))
        self.threshold_g = float(threshold_g)

    def zero(self):
        if self.state is None:
            raise RuntimeError("첫 센서값을 받은 뒤 화면 영점을 누르세요.")
        self.zero_fz = self.state[2]
        self.window.clear()

    def update(self, values):
        values = tuple(float(value) for value in values)
        self.state = (values if self.state is None else
                      tuple(old + self.alpha * (new - old)
                            for old, new in zip(self.state, values)))
        mass_g = (self.state[2] - self.zero_fz) / GRAVITY * 1000
        self.window.append(mass_g)
        stable = (len(self.window) == self.window.maxlen and
                  max(self.window) - min(self.window) <= self.threshold_g)
        return self.state, mass_g, statistics.fmean(self.window) if stable else None


class App(tk.Tk):
    def __init__(self, host, port, tare_path):
        super().__init__()
        self.title("PIVOT · RB5 AFT200 + Robotiq 2F-85")
        self.geometry("780x610")
        self.protocol("WM_DELETE_WINDOW", self.shutdown)
        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont"):
            font.nametofont(name).configure(family="Noto Sans CJK KR", size=10)

        self.host = host
        self.tare_path = tare_path
        self.samples = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.display_filter = DisplayFilter()
        self.gripper = Robotiq2F85(port)
        self.gripper_keepalive_started = False
        self.sensor_status = tk.StringVar(value="연결 중")
        self.pivot_tare_status = tk.StringVar()
        self.gripper_status = tk.StringVar(value="연결 안 됨")
        self.mass = tk.StringVar(value="화면 영점을 누르세요")
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
        sensor = ttk.LabelFrame(self, text="AIDIN AFT200 · RB5 Modbus TCP")
        sensor.pack(fill="x", padx=12, pady=10)
        ttk.Label(sensor, text=f"제어박스 IP: {self.host}").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=6, pady=6)
        ttk.Label(sensor, textvariable=self.sensor_status).grid(row=0, column=2, padx=12)
        ttk.Button(sensor, text="화면용 Fz 영점", command=self.zero_display).grid(
            row=0, column=3, padx=6)

        ttk.Label(sensor, text="축").grid(row=1, column=0)
        ttk.Label(sensor, text="원값").grid(row=1, column=1)
        ttk.Label(sensor, text="필터값").grid(row=1, column=2)
        for row, (name, raw, filtered) in enumerate(
                zip(AXIS_LABELS, self.raw_vars, self.filtered_vars), 2):
            ttk.Label(sensor, text=name).grid(row=row, column=0, sticky="w", padx=6)
            ttk.Label(sensor, textvariable=raw, width=16).grid(row=row, column=1)
            ttk.Label(sensor, textvariable=filtered, width=16).grid(row=row, column=2)
        ttk.Label(sensor, textvariable=self.mass, font=("Noto Sans CJK KR", 13, "bold")).grid(
            row=8, column=0, columnspan=4, sticky="w", padx=6, pady=8)
        ttk.Label(sensor, textvariable=self.pivot_tare_status).grid(
            row=9, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Button(sensor, text="PIVOT 타어 확인", command=self.refresh_pivot_tare).grid(
            row=9, column=3, padx=6)

        grip = ttk.LabelFrame(self, text="Robotiq 2F-85 · USB/RS485")
        grip.pack(fill="x", padx=12, pady=10)
        self.port_box = ttk.Combobox(grip, textvariable=self.port, width=18)
        self.port_box.grid(row=0, column=0, padx=6, pady=8)
        ttk.Button(grip, text="새로고침", command=self.refresh_ports).grid(row=0, column=1)
        ttk.Button(grip, text="연결", command=self.connect_gripper).grid(row=0, column=2, padx=6)
        ttk.Label(grip, textvariable=self.gripper_status).grid(row=0, column=3, padx=8)

        ttk.Label(grip, text="속도").grid(row=1, column=0)
        ttk.Scale(grip, from_=1, to=255, variable=self.speed, orient="horizontal").grid(
            row=1, column=1, columnspan=2, sticky="ew")
        ttk.Label(grip, text="힘").grid(row=2, column=0)
        ttk.Scale(grip, from_=1, to=255, variable=self.force, orient="horizontal").grid(
            row=2, column=1, columnspan=2, sticky="ew")
        ttk.Button(grip, text="Reset + Activate", command=self.initialize_gripper).grid(
            row=3, column=0, padx=6, pady=12)
        ttk.Button(grip, text="열기", command=lambda: self.move_gripper(False)).grid(
            row=3, column=1, padx=6)
        ttk.Button(grip, text="닫기", command=lambda: self.move_gripper(True)).grid(
            row=3, column=2, padx=6)

        ttk.Label(self, text=(
            "화면용 Fz 영점은 간이 질량 표시만 0으로 만듭니다. "
            "PIVOT 6축 타어 파일은 바꾸지 않습니다."), foreground="#9b1c1c").pack(
                anchor="w", padx=18, pady=5)
        ttk.Label(self, text="그리퍼 동작 전 손과 물체를 손가락 사이에서 치우세요.",
                  foreground="#9b1c1c").pack(anchor="w", padx=18)

    def sensor_worker(self):
        while not self.stop_event.is_set():
            try:
                sensor = Aft200Sensor(self.host, hz=50.0)
                self.samples.put(("status", "연결됨"))
                for values in sensor.stream():
                    if self.stop_event.is_set():
                        return
                    self.samples.put(("sample", values))
            except Exception as error:
                self.samples.put(("status", f"오류: {error}"))
                self.stop_event.wait(1.0)

    def consume_samples(self):
        try:
            while True:
                kind, value = self.samples.get_nowait()
                if kind == "status":
                    self.sensor_status.set(value)
                    continue
                filtered, mass_g, stable_mass_g = self.display_filter.update(value)
                for variable, number in zip(self.raw_vars, value):
                    variable.set(f"{number:.3f}")
                for variable, number in zip(self.filtered_vars, filtered):
                    variable.set(f"{number:.3f}")
                self.mass.set(f"간이 질량 변화: {mass_g:+.1f} g" if stable_mass_g is None
                              else f"안정 질량 변화: {stable_mass_g:+.1f} g")
        except queue.Empty:
            pass
        if not self.stop_event.is_set():
            self.after(20, self.consume_samples)

    def zero_display(self):
        try:
            self.display_filter.zero()
            self.mass.set("화면 영점 완료 · 안정화 중")
        except RuntimeError as error:
            messagebox.showerror("화면 영점 실패", str(error))

    def refresh_pivot_tare(self):
        try:
            table = TareTable.load(self.tare_path)
            self.pivot_tare_status.set(
                f"PIVOT 6축 타어: {len(table.table)}/3 방향 · {self.tare_path}")
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
    App(args.host, args.port, args.tare).mainloop()


if __name__ == "__main__":
    main()
