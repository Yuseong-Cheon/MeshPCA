"""직렬 포트를 열지 않는 Robotiq 프로토콜 검사."""

import struct
import unittest

from pivot.robotiq import Robotiq2F85, modbus_crc
from pivot.rb5_ui import DisplayFilter


class RobotiqTest(unittest.TestCase):
    def test_known_crc(self):
        self.assertEqual(modbus_crc(bytes.fromhex("01030000000A")),
                         bytes.fromhex("C5CD"))

    def test_write_frame(self):
        gripper = Robotiq2F85()
        captured = []
        gripper.request = lambda payload, size: captured.append((payload, size)) or b""
        gripper._write(9, 255, 64, 32)
        self.assertEqual(captured, [(bytes.fromhex(
            "091003e8000306090000ff4020"), 8)])

    def test_status_decode(self):
        gripper = Robotiq2F85()
        body = bytes((9, 3, 6, 0x31, 0, 0, 255, 12, 7))
        response = body + modbus_crc(body)
        gripper.request = lambda _payload, _size: response
        status = gripper.status()
        self.assertTrue(status["activated"])
        self.assertEqual(status["position"], 12)
        self.assertEqual(status["fault"], 0)

    def test_move_waits_for_completion(self):
        gripper = Robotiq2F85()
        gripper._write = lambda *_args: None
        states = iter((
            {"fault": 0, "requested_position": 255, "object": 0},
            {"fault": 0, "requested_position": 255, "object": 3},
        ))
        gripper.status = lambda: next(states)
        self.assertEqual(gripper.move(255)["object"], 3)

    def test_display_filter(self):
        display = DisplayFilter(alpha=0.5)
        display.update([0, 0, 50, 0, 0, 0])
        self.assertAlmostEqual(display.update([0, 0, 48, 0, 0, 0])[2], 49.0)


if __name__ == "__main__":
    unittest.main()
