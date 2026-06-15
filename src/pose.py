from __future__ import annotations

import math

import cv2
import numpy as np


def camera_matrix_from_config(cfg: dict) -> np.ndarray:
    cam = cfg["camera"]
    return np.array(
        [
            [cam["fx"], 0.0, cam["cx"]],
            [0.0, cam["fy"], cam["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_marker_poses(
    corners: list[np.ndarray],
    ids: list[int],
    camera_matrix: np.ndarray,
    marker_length_m: float,
) -> list[dict]:
    if not ids:
        return []

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_length_m,
        camera_matrix,
        dist_coeffs,
    )

    results: list[dict] = []
    for marker_id, rvec, tvec, corner in zip(ids, rvecs, tvecs, corners):
        yaw_deg = _yaw_from_rvec(rvec)
        corner_pts = corner.reshape(4, 2).astype(int).tolist()
        tvec_mm = (tvec.reshape(3) * 1000.0).round(1).tolist()

        results.append(
            {
                "id": int(marker_id),
                "tvec_mm": tvec_mm,
                "yaw_deg": round(yaw_deg, 1),
                "corners": corner_pts,
            }
        )

    return results


def draw_axes(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    corners: list[np.ndarray],
    marker_length_m: float,
) -> np.ndarray:
    output = frame.copy()
    if not corners:
        return output

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_length_m,
        camera_matrix,
        dist_coeffs,
    )

    for rvec, tvec in zip(rvecs, tvecs):
        cv2.drawFrameAxes(output, camera_matrix, dist_coeffs, rvec, tvec, marker_length_m * 0.5)

    return output


def _yaw_from_rvec(rvec: np.ndarray) -> float:
    rot, _ = cv2.Rodrigues(rvec)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    return math.degrees(yaw)