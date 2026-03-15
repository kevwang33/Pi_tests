from gpiozero import Device, AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

# Configuration
Device.pin_factory = PiGPIOFactory()

# Servo configuration
SERVO_CONFIGS = {
    'servo1': {
        'pin': 18,
        'min_pulse_width': 0.0006,
        'max_pulse_width': 0.0023
    },
    # Ready for second servo
    # 'servo2': {
    #     'pin': 19,
    #     'min_pulse_width': 0.0006,
    #     'max_pulse_width': 0.0023
    # }
}

# Movement sequence (angles in degrees)
MOVEMENT_SEQUENCE = [90, 0, -90]
MOVE_DELAY = 2  # seconds between movements

# Initialize servos
servos = {}
for name, config in SERVO_CONFIGS.items():
    servos[name] = AngularServo(
        config['pin'], 
        min_pulse_width=config['min_pulse_width'],
        max_pulse_width=config['max_pulse_width']
    )

# Main control loop
try:
    while True:
        for angle in MOVEMENT_SEQUENCE:
            # Move all servos to the same angle
            for servo in servos.values():
                servo.angle = angle
            print(f"All servos moved to {angle}°")
            sleep(MOVE_DELAY)
            
except KeyboardInterrupt:
    print("\nStopping servos...")
    for servo in servos.values():
        servo.detach()  # Stop sending control signals