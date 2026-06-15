import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_config, process_image, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArUco localization on a still image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    telemetry, annotated = process_image(ROOT / args.image, cfg)
    save_outputs(telemetry, annotated, ROOT / args.out)

    print(json.dumps(telemetry, indent=2))
    print(f"\nSaved: {ROOT / args.out / 'telemetry.json'}")
    print(f"Saved: {ROOT / args.out / 'annotated.png'}")


if __name__ == "__main__":
    main()