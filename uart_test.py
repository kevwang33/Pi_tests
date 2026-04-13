import serial
import time

PORT = '/dev/ttyTHS1'
BAUD = 921600

ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Opened {PORT} @ {BAUD}")

# Test 1: Read bytes from PX4
print("\n--- Reading from PX4 (5 seconds) ---")
data = ser.read(500)
if data:
    print(f"Received {len(data)} bytes")
    print(f"Hex: {data[:50].hex()}")
    if data[0:1] == b'\xfd':
        print("First byte is 0xFD — valid MAVLink v2 sync byte ✓")
    elif data[0:1] == b'\xfe':
        print("First byte is 0xFE — valid MAVLink v1 sync byte ✓")
    else:
        print(f"First byte is 0x{data[0]:02x} — NOT a MAVLink sync byte ✗")
else:
    print("No data received — PX4 is not sending on this port")

# Test 2: Send a MAVLink v2 heartbeat to PX4
print("\n--- Sending heartbeat to PX4 (5 seconds) ---")
heartbeat = bytes([
    0xfd, 0x09, 0x00, 0x00, 0x00, 0xff, 0xbe, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x08,
    0x00, 0x00, 0x03, 0x52, 0x64
])
for i in range(50):
    ser.write(heartbeat)
    time.sleep(0.1)
print(f"Sent {50 * len(heartbeat)} bytes")
print("Now check PX4 nsh: mavlink status")
print("Look at instance #0 rx_message_count — if still 0, wiring is the problem")

ser.close()
print("\nDone.")
