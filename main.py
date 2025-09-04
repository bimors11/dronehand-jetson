import argparse
import asyncio
import json
import os
import pickle
import shutil
import tempfile
import threading
import time
import uuid
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from numpy.typing import NDArray

from detection import Output
from detector import Detector
from drone_manager import DroneManager, Telemetry
from stream_manager import StreamManager
from tracker import Tracker


class MissionManager:
    def __init__(self):
        self.rtsp_url: Optional[str] = None
        self.mission_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.tracker = Tracker()
        self.outputs: List[Output] = []
        self.processed_frame: NDArray[np.uint8] | None = None
        self.stream_enabled = False
        self.stream_thread: threading.Thread | None = None
        self.detector_enabled = False
        self.detector_thread: threading.Thread | None = None
        self.running_mission = False
        self.stream_manager: StreamManager | None = None
        self.frame: NDArray[np.uint8] | None = None
        self.frame_id: int = 0
        self.detector: Detector | None = None
        self.drone_manager: DroneManager | None = None
        self.drone_thread: threading.Thread | None = None
        self.ground_width: float = 36.0
        self.ground_height: float = 20.3
        self.mission_id: str | None = None

    def is_stream_enabled(self):
        with self.lock:
            return self.stream_enabled

    def is_detector_enabled(self):
        with self.lock:
            return self.detector_enabled

    def is_running_mission(self):
        with self.lock:
            return self.running_mission

    def set_breed(self, breed: str):
        model_path = f"models/{breed}"
        self.detector = Detector(model_path)

    def start_stream(self, source: str):
        with self.lock:
            if self.stream_enabled:
                self.end_stream()

            self.stream_manager = StreamManager(source)

            self.stream_enabled = True
            self.stream_thread = threading.Thread(target=self.process_stream, daemon=True)
            self.stream_thread.start()

    def end_stream(self):
        with self.lock:
            if not self.stream_enabled:
                raise ValueError("Stream is not running.")

            self.stream_enabled = False

            if self.stream_thread:
                self.stream_thread.join(timeout=5)
                self.stream_thread = None

            if self.stream_manager:
                self.stream_manager.stop()
                self.stream_manager = None

    def process_stream(self):
        while self.is_stream_enabled() and not self.is_detector_enabled():
            if not self.stream_manager:
                raise ValueError("Stream manager is not initialised.")

            ok, frame = self.stream_manager.read()

            if not ok:
                print("Error: Could not read frame from stream or stream has ended.")

                time.sleep(1)

                continue

            self.frame = frame

    def start_detector(self, breed: str):
        with self.lock:
            if self.detector_enabled:
                self.stop_detector()

            self.detector = Detector(f"models/{breed}")

            self.detector_enabled = True
            self.detector_thread = threading.Thread(target=self.detect_stream, daemon=True)
            self.detector_thread.start()

    def stop_detector(self):
        with self.lock:
            if not self.detector_enabled:
                raise ValueError("Detector is not running.")

            self.detector_enabled = False

            if self.detector_thread:
                self.detector_thread.join(timeout=5)
                self.detector_thread = None

    async def start_telemetry_loop(self):
        if not self.drone_manager:
            raise ValueError("Drone Manager is not initialised.")

        await self.drone_manager.connect()
        await self.drone_manager.update_telemetry()

    def run_drone(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.start_telemetry_loop())
        loop.close()

    async def connect_drone(self, mavlink_url: str):
        drone_manager = DroneManager(mavlink_url)
        self.drone_manager = drone_manager

        self.drone_thread = threading.Thread(target=self.run_drone, daemon=True)
        self.drone_thread.start()

    def detect_stream(self):
        while self.is_detector_enabled():
            if not self.detector:
                raise ValueError("Detector is not initialised.")

            if not self.stream_manager:
                raise ValueError("Stream Manager is not initialised.")

            ok, frame = self.stream_manager.read()

            if not ok:
                print("Error: Could not read frame from stream or stream has ended.")

                time.sleep(1)

                continue

            telemetry: Telemetry | None = None

            if self.running_mission and self.drone_manager:
                telemetry = self.drone_manager.get_telemetry()

            inference_time = int(time.time())

            detections = self.detector.detect(frame)

            if self.running_mission and telemetry:
                output = Output(detections, inference_time, self.frame_id)

                self.outputs.append(output)
                tracked_detections = self.tracker.update(detections, self.detector.labels, telemetry, self.ground_width, self.ground_height)

                with open(f"missions/{self.mission_id}/outputs/{self.frame_id}.pkl", "wb") as f:
                    pickle.dump(output, f)

                with open(f"missions/{self.mission_id}/outputs/{self.frame_id}-tracked.pkl", "wb") as f:
                    pickle.dump(tracked_detections, f)

                cv2.imwrite(f"missions/{self.mission_id}/frames/{self.frame_id}.jpg", frame)

                thumbs_dir = f"missions/{self.mission_id}/thumbnails"

                for detection in tracked_detections:
                    if os.path.exists(f"{thumbs_dir}/{detection.id}.jpg"):
                        continue

                    x1 = int(detection.box.x1)
                    y1 = int(detection.box.y1)
                    x2 = int(detection.box.x2)
                    y2 = int(detection.box.y2)

                    cv2.imwrite(f"{thumbs_dir}/{detection.id}.jpg", frame[y1:y2, x1:x2])

            processed_frame = frame.copy()

            # Draw bounding boxes onto the frame.
            for detection in detections:
                x1 = int(detection.box.x1)
                y1 = int(detection.box.y1)
                x2 = int(detection.box.x2)
                y2 = int(detection.box.y2)

                color = (129, 97, 233) if detection.label.endswith("not ok") else (86, 40, 233)

                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, detection.label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            self.frame = processed_frame

            self.frame_id += 1

    def start_mission(self, mission_id: str):
        with self.lock:
            if self.running_mission:
                raise ValueError("Mission is already running.")

            if not self.drone_manager:
                raise ValueError("Drone Manager is not initialised.")

            if not self.detector:
                raise ValueError("Detector is not initialised.")

            if not self.stream_manager:
                raise ValueError("Stream Manager is not initialised.")

            self.running_mission = True

        self.tracker = Tracker()
        self.mission_id = mission_id
        self.outputs = []
        self.frame_id = 0

        os.makedirs(f"missions/{mission_id}", exist_ok=True)
        os.makedirs(f"missions/{mission_id}/frames", exist_ok=True)
        os.makedirs(f"missions/{mission_id}/thumbnails", exist_ok=True)
        os.makedirs(f"missions/{mission_id}/outputs", exist_ok=True)

    def stop_mission(self):
        with self.lock:
            if not self.running_mission:
                raise ValueError("Mission is not running.")
            self.running_mission = False

        if self.mission_thread:
            self.mission_thread.join(timeout=5)
            self.mission_thread = None

    def get_stream(self, quality: int = 80, fps: int = 10):
        boundary = b"--frame"

        while True:
            if self.frame is None:
                continue

            ok, jpg = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            if not ok:
                continue

            yield (boundary + b"\r\n" + b"Content-Type: image/jpeg\r\n" + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg.tobytes() + b"\r\n")

            time.sleep(1.0 / max(fps, 1))

    def get_current_mission_stats(self):
        with self.lock:
            if not self.running_mission:
                return None

            detections = {
                "mission_id": self.mission_id,
                "frame_id": self.frame_id,
                "detections": [
                    {
                        "id": detection.id,
                        "label": detection.label,
                        "confidence": detection.confidence,
                        "latitude": detection.coordinates.latitude if detection.coordinates else None,
                        "longitude": detection.coordinates.longitude if detection.coordinates else None,
                    }
                    for detection in self.tracker.tracked_objects
                ],
            }

            with open(f"missions/{self.mission_id}/detections.json", "w") as f:
                json.dump(detections, f)

            return detections

    def update_settings(self, ground_width: float | None = None, ground_height: float | None = None):
        with self.lock:
            if ground_width:
                self.ground_width = ground_width
            if ground_height:
                self.ground_height = ground_height


def create_app(mission_manager: MissionManager) -> FastAPI:
    app = FastAPI()

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.post("/stream/start")
    async def start_stream(rtsp_url: str):
        try:
            mission_manager.start_stream(rtsp_url)

            return {"message": "Successfully started stream."}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/stream/end")
    async def end_stream():
        try:
            mission_manager.end_stream()

            return {"message": "Successfully ended stream."}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/detector/start")
    async def start_detector(breed: str):
        try:
            mission_manager.start_detector(breed)
            return {"message": "Successfully started Detector."}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/detector/stop")
    async def stop_detector():
        try:
            mission_manager.stop_detector()
            return {"message": "Successfully stopped Detector."}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/missions/start")
    async def start_mission():
        try:
            mission_id = str(uuid.uuid4())
            mission_manager.start_mission(mission_id)
            return {"message": "Mission started successfully.", "mission_id": mission_id}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/missions/end")
    async def end_mission():
        try:
            mission_manager.stop_mission()
            return {"message": "Mission stopped successfully."}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/drone/connect")
    async def connect_drone(mavlink_url: str):
        try:
            await mission_manager.connect_drone(mavlink_url)
            return {"message": "Successfully connected to drone."}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/missions/stats")
    async def get_mission_stats():
        return mission_manager.get_current_mission_stats()

    @app.get("/missions/{mission_id}/stats")
    async def get_mission_stats(mission_id: str):
        if not os.path.exists(f"missions/{mission_id}/detections.json"):
            raise HTTPException(status_code=404, detail="Mission not found.")

        with open(f"missions/{mission_id}/detections.json", "r") as f:
            return json.load(f)

    @app.get("/missions/{mission_id}/detections/{detection_id}")
    async def get_detection(mission_id: str, detection_id: str):
        return FileResponse(f"missions/{mission_id}/thumbnails/{detection_id}.jpg")

    @app.get("/missions/{mission_id}")
    async def get_mission_archive(mission_id: str, background_tasks: BackgroundTasks):
        directory_to_zip = f"missions/{mission_id}"
        if not os.path.isdir(directory_to_zip):
            raise HTTPException(status_code=404, detail="Mission not found.")

        tmpdir = tempfile.mkdtemp()
        zip_path_base = os.path.join(tmpdir, mission_id)

        try:
            zip_path = shutil.make_archive(zip_path_base, "zip", directory_to_zip)
        except Exception as e:
            shutil.rmtree(tmpdir)
            raise HTTPException(status_code=500, detail=f"Failed to create zip archive: {e}")

        background_tasks.add_task(shutil.rmtree, tmpdir)

        return FileResponse(
            path=zip_path,
            filename=f"mission_{mission_id}.zip",
            media_type="application/zip",
        )

    @app.get("/stream")
    def stream():
        return StreamingResponse(mission_manager.get_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "Service is running"}

    @app.post("/missions/settings")
    async def update_mission_settings(ground_width: float | None = None, ground_height: float | None = None):
        try:
            mission_manager.update_settings(ground_width, ground_height)
            return {"message": "Mission settings updated successfully."}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    app.mount("/", StaticFiles(directory="web", html=True), name="static")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server.")
    parser.add_argument("--rtsp-url", type=str, default="rtsp://0.0.0.0:8554/video1", help="RTSP URL of the video stream.")
    parser.add_argument("--mavlink-url", type=str, default="udp://0.0.0.0:14445", help="MAVLink URL of the telemetry server.")
    parser.add_argument("--breed", type=str, default="sheep", help="Breed of animal to track.")

    args = parser.parse_args()

    mission_manager = MissionManager()
    app = create_app(mission_manager)

    uvicorn.run(app, host=args.host, port=args.port)
