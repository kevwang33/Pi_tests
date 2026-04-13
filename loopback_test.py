import serial
import time

PORT = '/dev/ttyTHS1'
BAUD = 921600

ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Loopback test on {PORT} @ {BAUD}")
print("Make sure TX and RX are shorted together!\n")

test_data = bytes(range(256)) * 4  # 1024 bytes, all values 0x00-0xFF
ser.reset_input_buffer()

ser.write(test_data)
time.sleep(0.5)
received = ser.read(len(test_data))

print(f"Sent:     {len(test_data)} bytes")
print(f"Received: {len(received)} bytes")

if len(received) == 0:
    print("\nNo data received — TX and RX are not connected")
elif received == test_data:
    print("\nAll bytes match — Jetson UART is working perfectly ✓")
    print("Problem is the WIRE or FC connector, not the Jetson")
else:
    errors = sum(1 for a, b in zip(test_data, received) if a != b)
    print(f"\nData CORRUPTED — {errors} bytes differ out of {len(received)}")
    print("Jetson UART has a problem at this baud rate")
    print("\nFirst mismatch:")
    for i, (a, b) in enumerate(zip(test_data, received)):
        if a != b:
            print(f"  Byte {i}: sent 0x{a:02x}, got 0x{b:02x}")
            break

ser.close()
print("\nDone.")
