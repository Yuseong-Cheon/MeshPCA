"""Linux에서 쓰는 Robotiq 2F-85 USB/RS485 Modbus-RTU 드라이버."""

import argparse
import os
import select
import struct
import termios
import threading
import time


def modbus_crc(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return struct.pack("<H", crc)


class Robotiq2F85:
    """115200 8N1, Modbus slave 9 직렬 제어."""

    def __init__(self, port="/dev/ttyUSB0", slave=9, timeout_s=0.5):
        self.port = port
        self.slave = int(slave)
        self.timeout_s = float(timeout_s)
        self.fd = None
        self.lock = threading.Lock()

    @property
    def connected(self):
        return self.fd is not None

    def connect(self):
        self.close()
        try:
            self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_SYNC)
            attrs = termios.tcgetattr(self.fd)
            attrs[0] = attrs[1] = attrs[3] = 0
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[4] = attrs[5] = termios.B115200
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except PermissionError as error:
            self.close()
            raise PermissionError(
                f"{self.port} 권한이 없습니다. dialout 그룹 또는 setfacl 설정을 확인하세요.") from error
        except Exception:
            self.close()
            raise
        return self.status()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _read(self, expected_size):
        response = b""
        deadline = time.monotonic() + self.timeout_s
        while len(response) < expected_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.fd], [], [], remaining)[0]:
                break
            response += os.read(self.fd, expected_size - len(response))
            if len(response) >= 5 and response[1] & 0x80:
                break
        return response

    def request(self, payload, response_size):
        if not self.connected:
            raise RuntimeError("Robotiq 포트가 연결되지 않았습니다.")
        with self.lock:
            termios.tcflush(self.fd, termios.TCIFLUSH)
            os.write(self.fd, payload + modbus_crc(payload))
            response = self._read(response_size)
        if len(response) < 5:
            raise TimeoutError("Robotiq Modbus 응답이 없습니다.")
        if modbus_crc(response[:-2]) != response[-2:]:
            raise RuntimeError("Robotiq 응답 CRC가 잘못됐습니다.")
        if response[0] != self.slave or response[1] & 0x7F != payload[1]:
            raise RuntimeError("예상하지 못한 Robotiq 응답입니다.")
        if response[1] & 0x80:
            raise RuntimeError(f"Robotiq Modbus 예외 코드: {response[2]}")
        if len(response) != response_size:
            raise RuntimeError(f"Robotiq 응답 길이 오류: {len(response)}/{response_size}")
        return response

    def _write(self, action, position=0, speed=64, force=32):
        values = (position, speed, force)
        if any(not 0 <= int(value) <= 255 for value in values):
            raise ValueError("position, speed, force는 0..255여야 합니다.")
        payload = struct.pack(
            ">BBHHB6B", self.slave, 0x10, 0x03E8, 3, 6,
            action, 0, 0, int(position), int(speed), int(force))
        return self.request(payload, 8)

    def status(self):
        payload = struct.pack(">BBHH", self.slave, 3, 0x07D0, 3)
        response = self.request(payload, 11)
        flags = response[3]
        return {
            "activated": bool(flags & 0x01),
            "status": (flags >> 4) & 0x03,
            "object": (flags >> 6) & 0x03,
            "fault": response[5],
            "requested_position": response[6],
            "position": response[7],
            "current": response[8],
        }

    def initialize(self):
        self._write(0)
        time.sleep(0.3)
        self._write(1)
        return self._wait(lambda state: state["status"] == 3 and state["fault"] == 0)

    def _wait(self, done, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.status()
            if state["fault"] >= 0x0A:
                raise RuntimeError(f"Robotiq major fault: 0x{state['fault']:02X}")
            if done(state):
                return state
            time.sleep(0.1)
        raise TimeoutError("Robotiq 동작 완료를 기다리다 시간 초과했습니다.")

    def move(self, position, speed=64, force=32):
        self._write(9, position, speed, force)
        return self._wait(lambda state: state["requested_position"] == int(position)
                          and state["object"] != 0)

    def open(self, speed=64, force=32):
        return self.move(0, speed, force)

    def close_gripper(self, speed=64, force=32):
        return self.move(255, speed, force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "initialize", "open", "close"))
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--speed", type=int, default=64)
    parser.add_argument("--force", type=int, default=32)
    args = parser.parse_args()
    gripper = Robotiq2F85(args.port)
    try:
        status = gripper.connect()
        if args.command == "initialize":
            status = gripper.initialize()
        elif args.command == "open":
            status = gripper.open(args.speed, args.force)
        elif args.command == "close":
            status = gripper.close_gripper(args.speed, args.force)
        print(status)
    finally:
        gripper.close()


if __name__ == "__main__":
    main()
