import asyncio
import threading

from mavsdk import System


class Telemetry:
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    heading: float = 0.0


class DroneManager:
    def __init__(self, mavlink_url: str):
        self.drone = System()
        self.connected = False
        self.lock = threading.Lock()
        self.telemetry = Telemetry()

        system_address = mavlink_url

        if system_address.startswith("udp") or system_address.startswith("tcp"):
            protocol, ip = system_address.split("://")
            system_address = f"{protocol}in://{ip}"

        self.system_address = system_address

    def get_telemetry(self):
        with self.lock:
            return self.telemetry

    async def connect(self):
        await self.drone.connect(self.system_address)

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected to drone")
                self.connected = True
                break

    async def update_telemetry(self):
        await asyncio.gather(self.update_position())

    async def update_position(self):
        async for pos in self.drone.telemetry.position():
            with self.lock:
                self.telemetry.latitude = pos.latitude_deg
                self.telemetry.longitude = pos.longitude_deg
                self.telemetry.altitude = pos.relative_altitude_m

    async def update_heading(self):
        async for heading in self.drone.telemetry.heading():
            with self.lock:
                self.telemetry.heading = heading.heading_deg
