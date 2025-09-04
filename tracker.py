from byte_tracker import BYTETracker
from coordinates_mapper import get_object_coordinates
from detection import Box, Coordinates, Detection
from drone_manager import Telemetry


class Tracker:
    def __init__(self):
        self.byte_tracker = BYTETracker()
        self.last_velocity_x = 0.0
        self.last_velocity_y = 0.0
        self.last_gimbal_yaw = 0.0
        self.tracked_objects: list[Detection] = []

    def update(self, dets: list[Detection], labels: list[str], telemetry: Telemetry, ground_width: float, ground_height: float) -> list[Detection]:
        if not dets:
            return []

        dets_geo: list[Detection] = []

        for det in dets:
            coords = get_object_coordinates(ground_width, ground_height, det.box.cx, det.box.cy, telemetry)

            dets_geo.append(
                Detection(
                    id=det.id,
                    box=det.box,
                    label=det.label,
                    cls=det.cls,
                    confidence=det.confidence,
                    coordinates=Coordinates(coords[0], coords[1]),
                )
            )

        tracker_output = self.byte_tracker.update(dets_geo)

        tracked_objects: list[Detection] = []

        for det in tracker_output:
            cx = (det[0] + det[2]) / 2
            cy = (det[1] + det[3]) / 2

            coords = get_object_coordinates(ground_width, ground_height, cx, cy, telemetry)

            tracked_objects.append(
                Detection(
                    id=int(det[4]),
                    box=Box(det[0], det[1], det[2], det[3], cx, cy, det[2], det[3]),
                    label=labels[int(det[6])],
                    cls=det[6],
                    confidence=float(det[5]),
                    coordinates=Coordinates(float(coords[0]), float(coords[1])),
                )
            )

        for tracked_object in tracked_objects:
            existing = next((obj for obj in self.tracked_objects if obj.id == tracked_object.id), None)

            if existing:
                index = self.tracked_objects.index(existing)
                self.tracked_objects[index] = tracked_object
            else:
                self.tracked_objects.append(tracked_object)

        return self.tracked_objects
