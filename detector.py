from typing import List, cast

import cv2
import numpy as np
import tensorflow as tf
import yaml
from cv2.typing import MatLike
from numpy.typing import NDArray

from detection import Box, Detection


class Detector:
    def __init__(self, model_path: str):
        self.model = tf.saved_model.load(model_path)  # type: ignore
        self.infer = self.model.signatures["serving_default"]  # type: ignore
        self.output_tensor_name = cast(str, list(self.infer.structured_outputs.keys())[0])  # type: ignore
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        with open(f"{model_path}/metadata.yaml", "r") as f:
            self.labels = cast(list[str], list(yaml.safe_load(f)["names"].values()))

    def detect(self, frame: MatLike):
        input_tensor = self.preprocess(frame, 1024, 1024)
        outputs = self.infer(tf.constant(input_tensor))  # type: ignore
        predictions = cast(NDArray[np.float32], outputs[self.output_tensor_name].numpy())  # type: ignore

        return self.postprocess(frame, predictions)

    def preprocess(self, frame: MatLike, width: int, height: int) -> NDArray[np.float32]:
        # Resize and convert to RGB.
        adjusted = cv2.resize(frame, (width, height))
        adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)

        # Normalise.
        adjusted = adjusted.astype(np.float32) / 255.0

        return np.expand_dims(adjusted, axis=0)

    def postprocess(self, frame: MatLike, predictions: NDArray[np.float32]):
        frame_height, frame_width, _ = frame.shape

        preds = np.transpose(predictions, (0, 2, 1))

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []
        detections: List[Detection] = []

        for detection in preds[0]:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.confidence_threshold:
                cx, cy, w, h = detection[0:4]
                x_min = int((cx - w / 2) * frame_width)
                y_min = int((cy - h / 2) * frame_height)
                width = int(w * frame_width)
                height = int(h * frame_height)

                boxes.append([x_min, y_min, width, height])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        if len(indices) == 0:
            return detections

        for i in indices:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cls = min(class_ids[i] + 1, len(self.labels) - 1)
            label = self.labels[cls]

            detections.append(
                Detection(
                    id=i,
                    box=Box(x, y, x + w, y + h, x + w / 2, y + h / 2, w, h),
                    cls=cls,
                    label=label,
                    confidence=confidences[i],
                    # image=frame[y : y + h, x : x + w],
                )
            )

        return detections
