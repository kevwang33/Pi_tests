from pymavlink import mavutil
import time

# Connect to Pixhawk
print("Connecting...")
master = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)
master.wait_heartbeat()
print("Connected! Heartbeat from system %u component %u" % 
      (master.target_system, master.target_component))

def set_mode(mode):
    """Set flight mode (e.g., STABILIZE, OFFBOARD, MANUAL)"""
    mode_id = master.mode_mapping()[mode]
    master.set_mode(mode_id)
    print(f"Mode set to {mode}")

def arm():
    """Arm the drone"""
    master.arducopter_arm()
    master.motors_armed_wait()
    print("Armed!")

def disarm():
    """Disarm the drone"""
    master.arducopter_disarm()
    master.motors_disarmed_wait()
    print("Disarmed!")

def takeoff(altitude=2):
    """Takeoff to given altitude in meters"""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,       # confirmation
        0, 0, 0, 0, 0, 0,
        altitude  # target altitude
    )
    print(f"Taking off to {altitude}m...")

def land():
    """Land the drone"""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    print("Landing...")

def goto(lat, lon, alt):
    """Go to GPS position"""
    master.mav.set_position_target_global_int_send(
        0,  # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111000,  # type_mask (only positions)
        int(lat * 1e7),
        int(lon * 1e7),
        alt,
        0, 0, 0,  # velocity
        0, 0, 0,  # acceleration
        0, 0      # yaw, yaw_rate
    )
    print(f"Going to lat={lat}, lon={lon}, alt={alt}m")

def get_position():
    """Get current GPS position"""
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
    if msg:
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000
        print(f"Position: lat={lat}, lon={lon}, alt={alt}m")
        return lat, lon, alt
    return None

# ============================================
# TEST SEQUENCE (props off!)
# ============================================

# Step 1: Check position
get_position()

# Step 2: Arm
arm()
time.sleep(2)

# Step 3: Disarm
disarm()

# ----- ONLY DO BELOW OUTDOORS WITH PROPS ON AND RC READY -----
# set_mode("OFFBOARD")  # or "GUIDED" for ArduPilot
# arm()
# time.sleep(2)
# takeoff(2)
# time.sleep(10)
# land()
