import asyncio
from mavsdk import System

async def test():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyTHS1:921600")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    # Read battery (no GPS needed)
    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.voltage_v:.2f}V, remaining: {battery.remaining_percent:.0f}%")
        break

    # Read attitude (IMU only, no GPS needed)
    async for attitude in drone.telemetry.attitude_euler():
        print(f"Roll: {attitude.roll_deg:.1f}°, Pitch: {attitude.pitch_deg:.1f}°, Yaw: {attitude.yaw_deg:.1f}°")
        break

    # Read health to see what's available
    async for health in drone.telemetry.health():
        print(f"Accelerometer ok: {health.is_accelerometer_calibration_ok}")
        print(f"Gyroscope ok: {health.is_gyrometer_calibration_ok}")
        print(f"Magnetometer ok: {health.is_magnetometer_calibration_ok}")
        print(f"GPS ok: {health.is_global_position_ok}")
        print(f"Armable: {health.is_armable}")
        break

    # Try force-arm then immediately disarm (props off!)
    print("Attempting force-arm...")
    try:
        await drone.action.arm_force()
        print("Armed!")
        await asyncio.sleep(2)
        await drone.action.disarm()
        print("Disarmed!")
    except Exception as e:
        print(f"Arm failed: {e}")

asyncio.run(test())