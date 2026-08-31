"""AIDIN AFT200 Modbus reader and gravity-direction tare table."""

import json
from pathlib import Path
import socket
import struct
import time

import numpy as np


class Aft200Sensor:
    """Read holding registers 304--309 from the AFT200 controller."""

    def __init__(self, host="192.168.50.51", hz=50.0, timeout_s=2.0):
        self.host = host
        self.hz = float(hz)
        self.timeout_s = float(timeout_s)
        if self.hz <= 0:
            raise ValueError("sampling frequency must be positive")
        self.transaction_id = 1

    @staticmethod
    def _recv_exact(sock, size):
        data = b""
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("AFT200 controller closed the connection")
            data += chunk
        return data

    def _read_one(self, sock):
        transaction_id = self.transaction_id
        sock.sendall(struct.pack(">HHHBBHH", transaction_id, 0, 6, 1, 3, 304, 6))
        header = self._recv_exact(sock, 7)
        response_id, protocol_id, length, unit_id = struct.unpack(">HHHB", header)
        body = self._recv_exact(sock, length - 1)
        expected = (transaction_id, 0, 1, b"\x03\x0c")
        if (response_id, protocol_id, unit_id, body[:2]) != expected:
            raise RuntimeError("unexpected AFT200 Modbus response: " + (header + body).hex())
        self.transaction_id = transaction_id % 65535 + 1
        return np.asarray(struct.unpack(">6h", body[2:14]), dtype=float) * 0.02

    def read_raw(self, n_samples):
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("sample count must be positive")
        samples = []
        next_sample = time.monotonic()
        with socket.create_connection((self.host, 502), timeout=self.timeout_s) as sock:
            sock.settimeout(self.timeout_s)
            for _ in range(n_samples):
                samples.append(self._read_one(sock))
                next_sample += 1.0 / self.hz
                time.sleep(max(0.0, next_sample - time.monotonic()))
        return np.mean(samples, axis=0)

    def stream(self):
        """Keep one TCP connection open for a live monitor."""
        next_sample = time.monotonic()
        with socket.create_connection((self.host, 502), timeout=self.timeout_s) as sock:
            sock.settimeout(self.timeout_s)
            while True:
                yield self._read_one(sock)
                next_sample += 1.0 / self.hz
                time.sleep(max(0.0, next_sample - time.monotonic()))


class TareTable:
    """Empty-tool wrench indexed by gravity direction."""

    def __init__(self):
        self.table = {}

    @staticmethod
    def key(g_hat):
        g_hat = np.asarray(g_hat, dtype=float)
        if g_hat.shape != (3,) or not np.isfinite(g_hat).all():
            raise ValueError("gravity direction must be a finite 3-vector")
        return tuple(np.round(g_hat, 6))

    def record(self, g_hat, wrench):
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("wrench must be a finite 6-vector")
        self.table[self.key(g_hat)] = wrench

    def apply(self, g_hat, wrench):
        key = self.key(g_hat)
        if key not in self.table:
            raise KeyError(f"missing tare for gravity direction {key}")
        return np.asarray(wrench, dtype=float) - self.table[key]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = [{"g_hat": list(g), "wrench": wrench.tolist()}
                   for g, wrench in self.table.items()]
        path.write_text(json.dumps({"entries": entries}, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        table = cls()
        for entry in json.loads(Path(path).read_text())["entries"]:
            table.record(entry["g_hat"], entry["wrench"])
        return table


def stable_wrench(sensor, samples, force_delta_n=0.5, torque_delta_nm=0.05):
    """Average two batches only when the stationary readings agree."""
    first = np.asarray(sensor.read_raw(samples), dtype=float)
    second = np.asarray(sensor.read_raw(samples), dtype=float)
    if first.shape != (6,) or second.shape != (6,):
        raise ValueError("sensor must return a 6-vector")
    delta = np.abs(second - first)
    if np.max(delta[:3]) > force_delta_n or np.max(delta[3:]) > torque_delta_nm:
        raise RuntimeError(f"AFT200 is not stable: delta={np.round(delta, 4)}")
    return (first + second) / 2.0
