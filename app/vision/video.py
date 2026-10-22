from __future__ import annotations

import math
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import Settings


class VideoInputError(ValueError):
    """The uploaded file is not a supported, bounded video clip."""


class VisionUnavailable(RuntimeError):
    """The optional CV runtime or configured model is unavailable."""


_MODEL_LOCK = Lock()


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> Any:
    with _MODEL_LOCK:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise VisionUnavailable("Install optional CV dependencies with: pip install -r requirements-vision.txt") from exc
        try:
            return YOLO(model_name)
        except Exception as exc:
            raise VisionUnavailable("The configured YOLO weights could not be loaded") from exc


def analyze_video(path: Path, settings: Settings) -> dict[str, Any]:
    """Sample a short clip and return object detections without retaining frames."""
    try:
        import cv2
    except ImportError as exc:
        raise VisionUnavailable("Install optional CV dependencies with: pip install -r requirements-vision.txt") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoInputError("Не удалось декодировать видео; поддерживаются локально декодируемые MP4/MOV/MKV/WEBM/AVI")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise VideoInputError("В видео отсутствуют корректные метаданные кадров")
        duration = frame_count / fps
        if duration > settings.vision_max_duration_seconds:
            raise VideoInputError(f"Длительность превышает лимит {settings.vision_max_duration_seconds} секунд")
        if max(width, height) > settings.vision_max_dimension * 4:
            raise VideoInputError(f"Размер кадра превышает поддерживаемый предел {settings.vision_max_dimension * 4} пикселей")

        model = _load_model(settings.vision_model)
        frames: list[dict[str, Any]] = []
        class_counts: Counter[str] = Counter()
        inference_started = time.perf_counter()
        model_time = 0.0
        timestamp = 0.0
        while timestamp < duration and len(frames) < settings.vision_max_frames:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                # Some containers do not support accurate seeking. Skip unreadable samples.
                timestamp += settings.vision_sample_interval_seconds
                continue
            original_height, original_width = frame.shape[:2]
            scale = min(1.0, settings.vision_max_dimension / max(original_width, original_height))
            if scale < 1.0:
                frame = cv2.resize(frame, (max(1, round(original_width * scale)), max(1, round(original_height * scale))))
            one_started = time.perf_counter()
            try:
                with _MODEL_LOCK:
                    results = model.predict(source=frame, conf=settings.vision_confidence, verbose=False)
            except Exception as exc:
                raise VisionUnavailable("YOLO inference failed; check the model file, device, and video codec") from exc
            model_time += time.perf_counter() - one_started
            result = results[0] if results else None
            detections: list[dict[str, Any]] = []
            if result is not None and result.boxes is not None:
                names = result.names
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
                    confidence = float(box.conf[0].item())
                    coordinates = [round(float(value) / scale, 2) for value in box.xyxy[0].tolist()]
                    detections.append({"class_name": class_name, "confidence": round(confidence, 4), "bbox_xyxy": coordinates})
                    class_counts[class_name] += 1
            frames.append({"timestamp_seconds": round(timestamp, 3), "detections": detections})
            timestamp += settings.vision_sample_interval_seconds

        if not frames:
            raise VideoInputError("Не удалось прочитать кадры из видео")
        warnings = [
            "Подсчёты отражают детекции на выбранных кадрах, а не количество уникальных объектов.",
            "COCO-модель общего назначения не обучена на материалах Nurtaro и не оценивает развитие, эмоции или безопасность ребёнка.",
        ]
        if duration / settings.vision_sample_interval_seconds > len(frames):
            warnings.append("Обработана только часть временных точек из-за лимита кадров или особенностей декодера.")
        return {
            "model": settings.vision_model,
            "duration_seconds": round(duration, 3),
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "sampled_frames": len(frames),
            "detection_counts_by_class": dict(sorted(class_counts.items())),
            "frames": frames,
            "inference_ms": round(model_time * 1000, 2),
            "warnings": warnings,
            "processing_ms": round((time.perf_counter() - inference_started) * 1000, 2),
        }
    finally:
        capture.release()
