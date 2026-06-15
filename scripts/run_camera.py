import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_config, process_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArUco localization on webcam feed")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        telemetry, annotated = process_frame(frame, cfg)
        print(json.dumps(telemetry))

        cv2.imshow("aruco-localizer", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()