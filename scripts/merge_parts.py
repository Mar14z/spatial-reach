"""合并并行生成的 part 目录为 train/val。

用法：
    python scripts/merge_parts.py --base data/synthetic/m0_kp9
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def merge(parts: Path, prefix: str, base: Path, out_name: str) -> int:
    out = base / out_name
    out.mkdir(parents=True, exist_ok=True)
    frame_id = 0
    for part in sorted(parts.glob(f"{prefix}*")):
        for frame in sorted(p for p in part.iterdir() if p.is_dir()):
            dst = out / f"{frame_id:06d}"
            shutil.move(str(frame), str(dst))
            meta_path = dst / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["frame"] = frame_id
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            frame_id += 1
    return frame_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path, default=ROOT / "data/synthetic/m0_kp9"
    )
    args = parser.parse_args()
    parts = args.base / "parts"

    n_train = merge(parts, "p", args.base, "train")
    n_val = merge(parts, "v", args.base, "val")
    print(f"train: {n_train} frames")
    print(f"val:   {n_val} frames")
    for part in sorted(parts.glob("*")):
        if part.is_dir() and not any(part.iterdir()):
            part.rmdir()
    if parts.exists() and not any(parts.iterdir()):
        parts.rmdir()


if __name__ == "__main__":
    main()
