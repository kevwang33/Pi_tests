from pymavlink import mavutil
import time
import threading
import os

SRC_SYSTEM = 255
SRC_COMPONENT = 190

SERIAL_PORT = os.environ.get('MAV_PORT', '/dev/ttyTHS1')
BAUD = int(os.environ.get('MAV_BAUD', '57600'))

send_lock = threading.Lock()

print(f"Connecting on {SERIAL_PORT} @ {BAUD} …")
master = mavutil.mavlink_connection(
    SERIAL_PORT, baud=BAUD,
    source_system=SRC_SYSTEM, source_component=SRC_COMPONENT,
)

# Force MAVLink 2 framing (PX4 defaults to v2 on most ports)
master.mav.WIRE_PROTOCOL_VERSION = '2.0'
os.environ['MAVLINK20'] = '1'

master.wait_heartbeat()
print("Connected! Heartbeat from system %u component %u" %
      (master.target_system, master.target_component))

if master.target_component != 1:
    print(f"Overriding target_component {master.target_component} → 1 (autopilot)")
    master.target_component = 1

print(f"GCS identity:  sys={SRC_SYSTEM}  comp={SRC_COMPONENT}")
print(f"Target:        sys={master.target_system}  comp={master.target_component}")
print(f"MAVLink wire:  {master.mav.WIRE_PROTOCOL_VERSION}")

# ------------------------------------------------------------------
# Background GCS heartbeat (thread-safe)
# ------------------------------------------------------------------
_hb_stop = threading.Event()

def _heartbeat_loop():
    while not _hb_stop.is_set():
        with send_lock:
            master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0,
            )
        time.sleep(1)

_hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
_hb_thread.start()

print("Sending GCS heartbeats …")
time.sleep(3)

# ------------------------------------------------------------------
# Bidirectional link test — request a single parameter
# ------------------------------------------------------------------
print("\n--- Link Test: requesting param SYS_AUTOSTART ---")
flush_start = time.time()
while master.recv_match(blocking=False):
    pass
print(f"  buffer flushed in {time.time() - flush_start:.2f}s")

with send_lock:
    master.mav.param_request_read_send(
        master.target_system, master.target_component,
        b'SYS_AUTOSTART', -1,
    )

param = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=5)
if param:
    print(f"  ✅ Link OK — {param.param_id.rstrip(chr(0))} = {param.param_value}")
else:
    print("  ❌ No PARAM_VALUE response!")
    print()
    print("  PX4 is NOT responding to messages from the Jetson.")
    print("  Likely causes:")
    print("    1) TX wiring: Jetson TX is not connected to Pixhawk RX")
    print("    2) Wrong port: try  MAV_PORT=/dev/ttyTHS2 python basic_flight.py")
    print("    3) Baud mismatch: try  MAV_BAUD=921600 python basic_flight.py")
    print("    4) PX4 serial port not configured for MAVLink")
    print("       → in QGC set MAV_1_CONFIG (or MAV_2_CONFIG) to the TELEM port wired to the Jetson")
    print("    5) Permission issue: run  sudo chmod 666 /dev/ttyTHS1")
    print()
    print("  Continuing anyway to show what happens …")
    print()

# ------------------------------------------------------------------
print("\nAvailable modes:")
print(list(master.mode_mapping().keys()))

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def flush_buffer():
    while master.recv_match(blocking=False):
        pass

def wait_for_ack(cmd_id, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=0.5)
        if msg is None:
            continue
        print(f"  [ACK] command={msg.command} result={msg.result}")
        if msg.command == cmd_id:
            return msg
    return None

def send_command_long(cmd_id, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0,
                      retries=3, timeout=5):
    for attempt in range(retries):
        flush_buffer()
        with send_lock:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                cmd_id,
                attempt,
                p1, p2, p3, p4, p5, p6, p7,
            )
        ack = wait_for_ack(cmd_id, timeout=timeout)
        if ack is not None:
            return ack
        print(f"  (no ACK – attempt {attempt + 1}/{retries})")
    return None

def set_mode(mode_name):
    if mode_name not in master.mode_mapping():
        print(f"Mode '{mode_name}' not found!")
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
        print(f"Mode result: {ack.result} ({'OK' if ack.result == 0 else 'FAIL'})")
    else:
        print("⚠  No ACK for set_mode")
    hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
    if hb:
        print(f"Current base_mode: {hb.base_mode}, custom_mode: {hb.custom_mode}")
    return True

def arm(force=True):
    ack = send_command_long(
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        p1=1, p2=21196 if force else 0,
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
    ack = send_command_long(
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        p1=0, p2=21196 if force else 0,
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
    ack = send_command_long(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, p7=altitude)
    if ack:
        if ack.result == 0:
            print(f"✅ TAKEOFF to {altitude}m ACCEPTED!")
        else:
            print(f"❌ TAKEOFF REJECTED (result={ack.result})")
    else:
        print("❌ No ACK received (timeout)")

def land():
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
    print("\n--- Setting Mode: OFFBOARD ---")
    set_mode("OFFBOARD")

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
