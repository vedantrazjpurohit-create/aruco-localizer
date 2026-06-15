import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import detect_markers, get_dictionary
from src.pipeline import load_config, process_image


def _marker_frame(dictionary) -> np.ndarray:
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    border = 80
    frame = np.full((marker.shape[0] + border * 2, marker.shape[1] + border * 2, 3), 255, dtype=np.uint8)
    frame[border : border + marker.shape[0], border : border + marker.shape[1], 0] = marker
    return frame


def test_detect_generated_marker():
    dictionary = get_dictionary("DICT_4X4_50")
    frame = _marker_frame(dictionary)

    corners, ids, _ = detect_markers(frame, dictionary)
    assert ids == [0]
    assert len(corners) == 1


def test_pipeline_on_sample_image():
    sample = ROOT / "data" / "sample_marker.png"
    if not sample.exists():
        import scripts.generate_sample as gen

        gen.main()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    telemetry, annotated = process_image(sample, cfg)

    assert telemetry["count"] == 1
    assert telemetry["markers"][0]["id"] == 0
    assert annotated.shape[0] > 0