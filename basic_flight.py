from pymavlink import mavutil
import time
import threading

# Identify ourselves as a GCS: system 255, component 190 (MAV_COMP_ID_MISSIONPLANNER)
SRC_SYSTEM = 255
SRC_COMPONENT = 190

print("Connecting...")
master = mavutil.mavlink_connection(
    '/dev/ttyTHS1', baud=57600,
    source_system=SRC_SYSTEM, source_component=SRC_COMPONENT,
)
master.wait_heartbeat()
print("Connected! Heartbeat from system %u component %u" %
      (master.target_system, master.target_component))

if master.target_component != 1:
    print(f"Overriding target_component {master.target_component} → 1 (autopilot)")
    master.target_component = 1

print(f"GCS identity: system={SRC_SYSTEM} component={SRC_COMPONENT}")
print(f"Target:       system={master.target_system} component={master.target_component}")

# Background thread: send GCS heartbeats so PX4 recognises us as a command source
_hb_stop = threading.Event()

def _heartbeat_loop():
    while not _hb_stop.is_set():
        master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )
        time.sleep(1)

_hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
_hb_thread.start()

# Give PX4 a couple of heartbeats before we start commanding
print("Sending GCS heartbeats … waiting for PX4 to register us")
time.sleep(3)

print("\nAvailable modes:")
print(list(master.mode_mapping().keys()))

def flush_buffer():
    """Drain any queued incoming messages before sending a command."""
    while master.recv_match(blocking=False):
        pass

def wait_for_ack(cmd_id, timeout=10):
    """Wait for COMMAND_ACK matching cmd_id; summarise other traffic."""
    start = time.time()
    other_counts = {}
    while time.time() - start < timeout:
        msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=0.5)
        if msg is None:
            continue
        print(f"  [ACK] command={msg.command} result={msg.result}")
        if msg.command == cmd_id:
            return msg
    if other_counts:
        print(f"  (received {sum(other_counts.values())} other messages, no matching ACK)")
    return None

def send_command_long(cmd_id, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0,
                      retries=3, timeout=5):
    """Send COMMAND_LONG with retries (re-transmit using the confirmation field)."""
    for attempt in range(retries):
        flush_buffer()
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            cmd_id,
            attempt,          # confirmation counter (0, 1, 2 …)
            p1, p2, p3, p4, p5, p6, p7,
        )
        ack = wait_for_ack(cmd_id, timeout=timeout)
        if ack is not None:
            return ack
        print(f"  (retry {attempt + 1}/{retries})")
    return None

def set_mode(mode_name):
    """Set PX4 flight mode."""
    if mode_name not in master.mode_mapping():
        print(f"Mode '{mode_name}' not found!")
        print(f"Available: {list(master.mode_mapping().keys())}")
        return False
    mode_id = master.mode_mapping()[mode_name]
    if isinstance(mode_id, (list, tuple)):
        base_mode = mode_id[0]
        main_mode = mode_id[1] if len(mode_id) > 1 else 0
        sub_mode  = mode_id[2] if len(mode_id) > 2 else 0
    else:
        base_mode, main_mode, sub_mode = mode_id, 0, 0
    ack = send_command_long(
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        p1=float(base_mode), p2=float(main_mode), p3=float(sub_mode),
    )
    if ack:
        print(f"Mode set result: {ack.result} ({'OK' if ack.result == 0 else 'FAIL'})")
    else:
        print("⚠  No ACK for set_mode – checking heartbeat for actual mode")
    hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
    if hb:
        print(f"Current base_mode: {hb.base_mode}, custom_mode: {hb.custom_mode}")
    return True

def arm(force=True):
    """Arm the drone (PX4)."""
    ack = send_command_long(
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        p1=1,                        # 1 = arm
        p2=21196 if force else 0,    # magic number to bypass pre-flight checks
    )
    if ack:
        if ack.result == 0:
            print("✅ ARM ACCEPTED!")
            return True
        print(f"❌ ARM REJECTED (result={ack.result})")
        return False
    print("❌ No ACK received (timeout)")
    return False

def disarm(force=True):
    """Disarm the drone (PX4)."""
    ack = send_command_long(
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        p1=0,                        # 0 = disarm
        p2=21196 if force else 0,
    )
    if ack:
        if ack.result == 0:
            print("✅ DISARM ACCEPTED!")
            return True
        print(f"❌ DISARM REJECTED (result={ack.result})")
        return False
    print("❌ No ACK received (timeout)")
    return False

def takeoff(altitude=2):
    """Takeoff to given altitude."""
    ack = send_command_long(
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        p7=altitude,
    )
    if ack:
        if ack.result == 0:
            print(f"✅ TAKEOFF to {altitude}m ACCEPTED!")
        else:
            print(f"❌ TAKEOFF REJECTED (result={ack.result})")
    else:
        print("❌ No ACK received (timeout)")

def land():
    """Land the drone."""
    ack = send_command_long(mavutil.mavlink.MAV_CMD_NAV_LAND)
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
try:
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
finally:
    _hb_stop.set()
    _hb_thread.join(timeout=2)