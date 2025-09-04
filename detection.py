import numpy as np
from numpy.typing import NDArray


class Box:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, width: float, height: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height


class Coordinates:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude


class Detection:
    def __init__(
        self,
        id: int,
        box: Box,
        cls: int,
        label: str,
        confidence: float,
        coordinates: Coordinates | None = None,
        image: NDArray[np.uint8] | None = None,
    ):
        self.id = id
        self.box = box
        self.cls = cls
        self.label = label
        self.confidence = confidence
        self.coordinates = coordinates
        self.image = image


class Output:
    def __init__(self, results: list[Detection], inferenceTime: int, frameId: int):
        self.results = results
        self.inferenceTime = inferenceTime
        self.frameId = frameId
