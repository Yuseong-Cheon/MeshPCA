"""Checks that do not connect to or move real hardware."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pivot.aft_tare import Aft200Sensor, TareTable, stable_wrench


class FakeSocket:
    def __init__(self, response):
        self.response = response

    def sendall(self, _data):
        pass

    def recv(self, size):
        chunk, self.response = self.response[:size], self.response[size:]
        return chunk


class FakeSensor:
    def __init__(self, readings):
        self.readings = iter(readings)

    def read_raw(self, _samples):
        return next(self.readings)


class AftTareTest(unittest.TestCase):
    def test_modbus_decode(self):
        payload = struct.pack(">6h", 4, 1, -1, 0, 0, 0)
        response = struct.pack(">HHHB", 1, 0, 15, 1) + b"\x03\x0c" + payload
        np.testing.assert_allclose(Aft200Sensor()._read_one(FakeSocket(response)),
                                   [0.08, 0.02, -0.02, 0, 0, 0])

    def test_tare_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tare.json"
            table = TareTable()
            table.record([0, 0, -1], [1, 2, 3, 4, 5, 6])
            table.save(path)
            loaded = TareTable.load(path)
            np.testing.assert_allclose(loaded.apply(
                [0, 0, -1], [2, 4, 6, 8, 10, 12]), [1, 2, 3, 4, 5, 6])

    def test_stability_gate(self):
        sensor = FakeSensor(([1, 2, 3, 0.1, 0.2, 0.3],
                             [1.2, 2, 3, 0.1, 0.22, 0.3]))
        np.testing.assert_allclose(stable_wrench(sensor, 100),
                                   [1.1, 2, 3, 0.1, 0.21, 0.3])
        unstable = FakeSensor((np.zeros(6), [0.6, 0, 0, 0, 0, 0]))
        with self.assertRaises(RuntimeError):
            stable_wrench(unstable, 100)


if __name__ == "__main__":
    unittest.main()
