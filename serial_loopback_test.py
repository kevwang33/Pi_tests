"""
Serial loopback test for Jetson UART.

Usage:
  1. Disconnect the serial cable from the Pixhawk
  2. Short (jumper wire) the TX and RX pins together on the Jetson header
  3. Run:  python serial_loopback_test.py
  4. If the test passes, TX hardware is working and the problem is wiring to the Pixhawk

You can also specify the port:
  MAV_PORT=/dev/ttyTHS2 python serial_loopback_test.py
"""

import serial
import os
import time

PORT = os.environ.get('MAV_PORT', '/dev/ttyTHS1')
BAUD = int(os.environ.get('MAV_BAUD', '921600'))
TEST_MSG = b'LOOPBACK_OK_12345\n'

print(f"Loopback test on {PORT} @ {BAUD}")
print(f"Make sure TX and RX pins are shorted together!\n")

try:
    ser = serial.Serial(PORT, BAUD, timeout=2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.1)

    ser.write(TEST_MSG)
    ser.flush()
    time.sleep(0.2)

    rx = ser.read(len(TEST_MSG))
    ser.close()

    if rx == TEST_MSG:
        print(f"  TX sent:     {TEST_MSG}")
        print(f"  RX received: {rx}")
        print("  ✅ LOOPBACK PASS — TX hardware is working!")
        print("     → Problem is likely wiring to the Pixhawk or PX4 config.")
    elif len(rx) > 0:
        print(f"  TX sent:     {TEST_MSG}")
        print(f"  RX received: {rx}")
        print("  ⚠  Partial/garbled data — possible baud mismatch or noise.")
    else:
        print(f"  TX sent:     {TEST_MSG}")
        print(f"  RX received: (nothing)")
        print("  ❌ LOOPBACK FAIL — no data echoed back.")
        print("     Possible causes:")
        print("       - TX/RX pins not actually shorted together")
        print("       - TX pin not functioning (device tree issue)")
        print("       - Wrong port (try MAV_PORT=/dev/ttyTHS2)")

except serial.SerialException as e:
    print(f"  ❌ Could not open {PORT}: {e}")
