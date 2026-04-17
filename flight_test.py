#!/usr/bin/env python3
"""
PX4 Offboard flight script for Jetson Orin Nano + Holybro Pixhawk Jetson Baseboard.

Connects via serial (TELEM2 → /dev/ttyTHS1 @ 57600 baud), then:
  1. Waits for connection and valid local position estimate
  2. Sends initial offboard setpoint (hold position)
  3. Starts offboard mode
  4. Arms the vehicle
  5. Takes off to 3 m, yaws 90° (faces East)
  6. Lands and disarms
"""

import asyncio
import logging

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

logging.basicConfig(level=logging.INFO)

SERIAL_ADDRESS = "serial:///dev/ttyACM0:57600"
#SERIAL_ADDRESS = "serial:///dev/ttyTHS1:57600"
TAKEOFF_ALT = 3.0          # metres (NED ⇒ -3.0 m down)
YAW_TARGET = 90.0           # degrees – face East
CLIMB_SETTLE_S = 10         # seconds to wait after climb command
YAW_SETTLE_S = 5            # seconds to hold after yaw command
LAND_MONITOR_S = 0.5        # polling interval while waiting to touch down
HEALTH_TIMEOUT_S = 30       # max time to wait for valid position estimate


async def print_status_text(drone):
    """Background task that streams PX4 status messages."""
    try:
        async for status in drone.telemetry.status_text():
            print(f"[PX4] {status.type}: {status.text}")
    except asyncio.CancelledError:
        return


async def print_flight_mode(drone):
    """Background task that logs flight-mode transitions."""
    try:
        async for mode in drone.telemetry.flight_mode():
            print(f"[MODE] {mode}")
    except asyncio.CancelledError:
        return


async def wait_until_landed(drone, timeout=30):
    """Block until the vehicle reports it is no longer in the air."""
    elapsed = 0.0
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("-- Touchdown confirmed")
            return
        await asyncio.sleep(LAND_MONITOR_S)
        elapsed += LAND_MONITOR_S
        if elapsed >= timeout:
            print("-- Land timeout reached; proceeding anyway")
            return


async def wait_for_health(drone, timeout=HEALTH_TIMEOUT_S):
    """
    Wait until the EKF reports a usable local position estimate.
    Without this, PositionNedYaw setpoints are silently ignored.
    """
    print(f"-- Waiting for valid local position estimate (timeout {timeout}s)...")
    elapsed = 0.0
    async for health in drone.telemetry.health():
        ok_local = health.is_local_position_ok
        ok_home  = health.is_home_position_ok
        if ok_local and ok_home:
            print("-- Local position estimate OK, home position OK")
            return True
        if ok_local:
            print("-- Local position estimate OK (home not set yet)")
        await asyncio.sleep(1.0)
        elapsed += 1.0
        if elapsed >= timeout:
            print(f"!! TIMEOUT: local_position_ok={ok_local}, "
                  f"home_position_ok={ok_home}")
            print("!! The EKF has no valid position estimate. Offboard position "
                  "commands will NOT work. Check GPS lock or external position "
                  "source (MOCAP/optical flow).")
            return False


async def run():
    drone = System()
    await drone.connect(system_address=SERIAL_ADDRESS)

    status_task = asyncio.ensure_future(print_status_text(drone))
    mode_task = asyncio.ensure_future(print_flight_mode(drone))

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    # ── 0. Wait for a valid position estimate from the EKF ───────────
    health_ok = await wait_for_health(drone)
    if not health_ok:
        print("!! Aborting: cannot fly offboard-position without a position estimate.")
        status_task.cancel()
        mode_task.cancel()
        return

    # ── 1. Set an initial setpoint BEFORE starting offboard ──────────
    print("-- Setting initial offboard setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

    # ── 2. Start offboard mode ───────────────────────────────────────
    print("-- Starting offboard mode")
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        status_task.cancel()
        mode_task.cancel()
        return

    # ── 3. Arm ─────────────────────────────────────────────────────────
    print("-- Arming")
    await drone.action.arm()

    async for is_armed in drone.telemetry.armed():
        if is_armed:
            print("-- Armed confirmed")
            break

    # ── 4. Take off to 3 m (NED: z = -3) ────────────────────────────
    print(f"-- Climbing to {TAKEOFF_ALT} m")
    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, 0.0)
    )
    await asyncio.sleep(CLIMB_SETTLE_S)

    # ── 5. Yaw 90° while holding position ──────────────────────────
    print(f"-- Yawing to {YAW_TARGET}°")
    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, YAW_TARGET)
    )
    await asyncio.sleep(YAW_SETTLE_S)

    # ── 6. Land ──────────────────────────────────────────────────────
    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as e:
        print(f"Offboard stop failed: {e._result.result}")

    print("-- Landing")
    await drone.action.land()

    print("-- Waiting for touchdown...")
    await wait_until_landed(drone)

    # ── 7. Disarm ────────────────────────────────────────────────────
    print("-- Disarming")
    await drone.action.disarm()

    print("-- Flight complete")
    status_task.cancel()
    mode_task.cancel()


if __name__ == "__main__":
    asyncio.run(run())
