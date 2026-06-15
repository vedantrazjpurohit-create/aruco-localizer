"""Generate a sample ArUco marker image for offline testing."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import get_dictionary


def main() -> None:
    dictionary = get_dictionary("DICT_4X4_50")
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    border = 80
    canvas = np.full((marker.shape[0] + border * 2, marker.shape[1] + border * 2, 3), 255, dtype=np.uint8)
    canvas[border : border + marker.shape[0], border : border + marker.shape[1], 0] = marker
    out = ROOT / "data" / "sample_marker.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()