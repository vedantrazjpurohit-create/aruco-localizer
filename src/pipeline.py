from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.detector import detect_markers, draw_detections, get_dictionary
from src.pose import camera_matrix_from_config, draw_axes, estimate_marker_poses, solve_marker_poses


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_frame(frame: np.ndarray, cfg: dict) -> tuple[dict, np.ndarray]:
    dictionary = get_dictionary(cfg.get("dictionary", "DICT_4X4_50"))
    corners, ids, _ = detect_markers(frame, dictionary)

    camera_matrix = camera_matrix_from_config(cfg)
    marker_length_m = cfg["marker_length_mm"] / 1000.0

    rvecs = tvecs = None
    if ids:
        rvecs, tvecs = solve_marker_poses(corners, camera_matrix, marker_length_m)

    markers = estimate_marker_poses(
        corners, ids, camera_matrix, marker_length_m, rvecs=rvecs, tvecs=tvecs
    )

    annotated = draw_detections(frame, corners, ids)
    if rvecs is not None and tvecs is not None:
        annotated = draw_axes(annotated, camera_matrix, rvecs, tvecs, marker_length_m)

    telemetry = {"markers": markers, "count": len(markers)}
    return telemetry, annotated


def process_image(image_path: str | Path, cfg: dict) -> tuple[dict, np.ndarray]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return process_frame(frame, cfg)


def save_outputs(telemetry: dict, annotated: np.ndarray, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "telemetry.json"
    img_path = out / "annotated.png"

    json_path.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    cv2.imwrite(str(img_path), annotated)