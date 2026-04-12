from pymavlink import mavutil
import time

# Connect to Pixhawk
print("Connecting...")
master = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)
master.wait_heartbeat()
print("Connected! Heartbeat from system %u component %u" %
      (master.target_system, master.target_component))

# PX4 autopilot is always component 1; override if wait_heartbeat picked up
# a broadcast (component 0) or a relay instead of the autopilot directly.
if master.target_component != 1:
    print(f"Overriding target_component {master.target_component} → 1 (autopilot)")
    master.target_component = 1

# Print available modes
print("\nAvailable modes:")
print(list(master.mode_mapping().keys()))

def flush_buffer():
    """Drain any queued incoming messages before sending a command."""
    while master.recv_match(blocking=False):
        pass

def wait_for_ack(timeout=10):
    """Wait for COMMAND_ACK; print any other messages received while waiting."""
    start = time.time()
    while time.time() - start < timeout:
        msg = master.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue
        if msg.get_type() == 'COMMAND_ACK':
            return msg
        # Show unexpected messages to help diagnose communication issues
        print(f"  [recv] {msg.get_type()}")
    return None

def set_mode(mode_name):
    """Set PX4 flight mode"""
    if mode_name not in master.mode_mapping():
        print(f"Mode '{mode_name}' not found!")
        print(f"Available: {list(master.mode_mapping().keys())}")
        return False
    mode_id = master.mode_mapping()[mode_name]
    # PX4 mode_mapping returns (base_mode, main_mode, sub_mode);
    # MAV_CMD_DO_SET_MODE param1=base_mode, param2=main_mode, param3=sub_mode
    if isinstance(mode_id, (list, tuple)):
        base_mode = mode_id[0]
        main_mode = mode_id[1] if len(mode_id) > 1 else 0
        sub_mode  = mode_id[2] if len(mode_id) > 2 else 0
    else:
        base_mode, main_mode, sub_mode = mode_id, 0, 0
    flush_buffer()
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        float(base_mode),
        float(main_mode),
        float(sub_mode),
        0.0, 0.0, 0.0, 0.0
    )
    print(f"Mode command sent: {mode_name}")
    time.sleep(1)
    # Check current mode
    hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
    if hb:
        print(f"Current base_mode: {hb.base_mode}, custom_mode: {hb.custom_mode}")
    return True

def arm():
    """Arm the drone (PX4)"""
    flush_buffer()
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,      # 1 = arm
        21196,  # 21196 = force arm (bypass preflight checks)
        0, 0, 0, 0, 0
    )
    ack = wait_for_ack(5)
    if ack:
        if ack.result == 0:
            print("✅ ARM ACCEPTED!")
            return True
        else:
            print(f"❌ ARM REJECTED (result={ack.result})")
            return False
    else:
        print("❌ No ACK received (timeout)")
        return False

def disarm():
    """Disarm the drone (PX4)"""
    flush_buffer()
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,      # 0 = disarm
        21196,  # 21196 = force disarm
        0, 0, 0, 0, 0
    )
    ack = wait_for_ack(5)
    if ack:
        if ack.result == 0:
            print("✅ DISARM ACCEPTED!")
            return True
        else:
            print(f"❌ DISARM REJECTED (result={ack.result})")
            return False
    else:
        print("❌ No ACK received (timeout)")
        return False

def takeoff(altitude=2):
    """Takeoff to given altitude"""
    flush_buffer()
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0,
        altitude
    )
    ack = wait_for_ack(5)
    if ack:
        if ack.result == 0:
            print(f"✅ TAKEOFF to {altitude}m ACCEPTED!")
        else:
            print(f"❌ TAKEOFF REJECTED (result={ack.result})")
    else:
        print("❌ No ACK received (timeout)")

def land():
    """Land the drone"""
    flush_buffer()
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    ack = wait_for_ack(5)
    if ack:
        if ack.result == 0:
            print("✅ LAND ACCEPTED!")
        else:
            print(f"❌ LAND REJECTED (result={ack.result})")
    else:
        print("❌ No ACK received (timeout)")

# ============================================
# TEST SEQUENCE
# ============================================

print("\n--- Setting Mode ---")
set_mode("MANUAL")

print("\n--- Arming (force) ---")
arm()

time.sleep(3)

print("\n--- Sending Takeoff ---")
takeoff(2)

time.sleep(3)

print("\n--- Disarming (force) ---")
disarm()

print("\n--- Test Complete ---")