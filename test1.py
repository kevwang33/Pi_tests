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

    # Set manual mode first (no GPS needed)
    await drone.action.set_flight_mode(1)  # MANUAL

    # Set manual mode first (no GPS needed)
    await drone.action.set_flight_mode(1)  # MANUAL

    print("Attempting force-arm...")
    await drone.action.arm_force()
    print("Armed!")

    await asyncio.sleep(3)

    await drone.action.disarm()
    print("Disarmed!")

asyncio.run(test())