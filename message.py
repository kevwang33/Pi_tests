import serial
import time

# Configure serial connection
port = '/dev/ttyACM0'
baud_rate = 9600  # Change this to match your Arduino's baud rate

try:
    # Open serial connection
    ser = serial.Serial(port, baud_rate, timeout=1)
    print(f"Connected to MCU on {port}")
    
    # Wait a moment for connection to establish
    time.sleep(2)
    
    while True:
        # Send "hi" to Arduino
        ser.write(b'hi\n')
        print("Sent: hi")
        
        # Wait 3 seconds
        time.sleep(3)
        
except serial.SerialException as e:
    print(f"Error connecting to Arduino: {e}")
except KeyboardInterrupt:
    print("\nProgram stopped by user")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed")