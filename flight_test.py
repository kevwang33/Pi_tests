#!/usr/bin/env python3
"""
PX4 Offboard flight script for Jetson Orin Nano + Holybro Pixhawk Jetson Baseboard.

Connects via serial (TELEM2 → /dev/ttyTHS1 @ 921600 baud), then:
  1. Waits for connection
  2. Force-arms (bypasses pre-arm checks)
  3. Enters offboard mode and takes off to 3 m
  4. Yaws 90° (faces East)
  5. Lands and disarms
"""

import asyncio
import logging

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

logging.basicConfig(level=logging.INFO)

SERIAL_ADDRESS = "serial:///dev/ttyTHS1:921600"
TAKEOFF_ALT = 3.0          # metres (NED ⇒ -3.0 m down)
YAW_TARGET = 90.0           # degrees – face East
CLIMB_SETTLE_S = 10         # seconds to wait after climb command
YAW_SETTLE_S = 5            # seconds to hold after yaw command
LAND_MONITOR_S = 0.5        # polling interval while waiting to touch down


async def print_status_text(drone):
    """Background task that streams PX4 status messages."""
    try:
        async for status in drone.telemetry.status_text():
            print(f"[PX4] {status.type}: {status.text}")
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


async def run():
    drone = System()
    await drone.connect(system_address=SERIAL_ADDRESS)

    status_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    # ── 1. Force-arm (bypasses pre-arm checks like GPS) ────────────
    print("-- Force-arming (bypassing pre-arm checks)")
    await drone.action.arm_force()

    # ── 2. Set an initial setpoint BEFORE starting offboard ──────────
    print("-- Setting initial offboard setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

    # ── 3. Start offboard mode ───────────────────────────────────────
    print("-- Starting offboard mode")
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        status_task.cancel()
        return

    # ── 4. Take off to 3 m (NED: z = -3) ────────────────────────────
    print(f"-- Climbing to {TAKEOFF_ALT} m")
    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, 0.0)
    )
    await asyncio.sleep(CLIMB_SETTLE_S)

    # ── 5. Yaw 90° while holding position ───────────────────────────
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


if __name__ == "__main__":
    asyncio.run(run())
