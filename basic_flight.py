from pymavlink import mavutil
import time

# Connect to Pixhawk
print("Connecting...")
master = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)
master.wait_heartbeat()
print("Connected! Heartbeat from system %u component %u" % 
      (master.target_system, master.target_component))

# --- Listen to messages for 10 seconds ---
print("\n--- Listening for messages (10 sec) ---\n")
start = time.time()
while time.time() - start < 10:
    msg = master.recv_match(blocking=True, timeout=1)
    if msg:
        print(f"MSG: {msg.get_type()}")

print("\n--- Done listening ---\n")

# --- Try to set mode ---
print("Setting mode to STABILIZE...")
try:
    mode_id = master.mode_mapping()['STABILIZE']
    master.set_mode(mode_id)
    print("Mode command sent!")
except Exception as e:
    print(f"Mode failed: {e}")

time.sleep(2)

# --- Try to arm ---
print("\nSending ARM command...")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1, 0, 0, 0, 0, 0, 0  # 1 = arm
)

# Wait for response with timeout
print("Waiting for ARM response...")
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if ack:
    print(f"ARM ACK: result = {ack.result}")
    if ack.result == 0:
        print("✅ ARM ACCEPTED!")
    else:
        print(f"❌ ARM REJECTED (code {ack.result})")
else:
    print("❌ No ACK received (timeout)")

time.sleep(3)

# --- Try to takeoff ---
print("\nSending TAKEOFF command (2m)...")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0,
    0, 0, 0, 0, 0, 0,
    2  # altitude in meters
)

ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if ack:
    print(f"TAKEOFF ACK: result = {ack.result}")
    if ack.result == 0:
        print("✅ TAKEOFF ACCEPTED!")
    else:
        print(f"❌ TAKEOFF REJECTED (code {ack.result})")
else:
    print("❌ No ACK received (timeout)")

time.sleep(3)

# --- Disarm ---
print("\nSending DISARM command...")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    0, 0, 0, 0, 0, 0, 0  # 0 = disarm
)

ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if ack:
    print(f"DISARM ACK: result = {ack.result}")
    if ack.result == 0:
        print("✅ DISARM ACCEPTED!")
    else:
        print(f"❌ DISARM REJECTED (code {ack.result})")
else:
    print("❌ No ACK received (timeout)")

print("\n--- Test complete ---")