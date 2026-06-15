from __future__ import annotations

import cv2
import numpy as np


def get_dictionary(name: str = "DICT_4X4_50") -> cv2.aruco.Dictionary:
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(
    frame: np.ndarray,
    dictionary: cv2.aruco.Dictionary,
    parameters: cv2.aruco.DetectorParameters | None = None,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray] | None]:
    if parameters is None:
        parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is None:
        return [], [], rejected

    return corners, ids.flatten().tolist(), rejected


def draw_detections(
    frame: np.ndarray,
    corners: list[np.ndarray],
    ids: list[int],
) -> np.ndarray:
    output = frame.copy()
    if ids:
        cv2.aruco.drawDetectedMarkers(output, corners, np.array(ids))
    return output