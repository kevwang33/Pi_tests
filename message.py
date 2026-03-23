import serial
import time

# Configure serial connection
port = '/dev/ttyACM0'
baud_rate = 115200  # Change this to match your Arduino's baud rate

try:
    # Open serial connection
    ser = serial.Serial(port, baud_rate, timeout=1)
    print(f"Connected to MCU on {port}")
    
    # Wait a moment for connection to establish
    time.sleep(2)
    
    while True:
        # Send "hi" to Arduino
        ser.write(b'led_r_on\n')
        print("Sent: led_r_on")
        
        # Wait 3 seconds
        time.sleep(1)
        
        ser.write(b'led_r_off\n')
        print("Sent: led_r_off")
        
        # Wait 3 seconds
        time.sleep(1)
        
except serial.SerialException as e:
    print(f"Error connecting to Arduino: {e}")
except KeyboardInterrupt:
    print("\nProgram stopped by user")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed")