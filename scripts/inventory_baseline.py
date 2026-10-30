"""A01 输入身份盘点：显式指定仓库内 glob，只读输入，不执行模型或实验。"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="相对仓库根的文件匹配式；空匹配记录缺失")
    args = parser.parse_args()
    files, groups = {}, {}
    for pattern in args.patterns:
        paths = sorted(p for p in ROOT.glob(pattern) if p.is_file())
        groups[pattern] = len(paths)
        for path in paths:
            name = path.resolve().relative_to(ROOT).as_posix()
            if name in files:
                continue
            before = path.stat()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"盘点期间输入发生变化：{name}")
            files[name] = {"bytes": before.st_size, "sha256": digest}
    git = {}
    for key, command in {"head": ["rev-parse", "HEAD"],
                         "status": ["status", "--porcelain=v1", "-uall"],
                         "diff_stat": ["diff", "HEAD", "--stat"],
                         "diff": ["diff", "HEAD", "--binary", "--no-ext-diff"]}.items():
        git[key] = subprocess.check_output(["git", "-C", str(ROOT), *command]).decode("utf-8")
    run = ROOT / "outputs/runs" / ("a01-inventory-" + uuid4().hex)
    data = {"schema_version": 1, "kind": "input_inventory", "run_id": run.name,
            "status": "inventory_only" if all(groups.values()) else "inventory_incomplete",
            "root": str(ROOT), "patterns": groups, "files": files, "source": git,
            "environment": {"conda": os.environ.get("CONDA_DEFAULT_ENV"),
                            "python": platform.python_version(), "platform": platform.platform(),
                            "packages": sorted((d.metadata["Name"], d.version) for d in distributions())}}
    run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"manifest": str(run / "manifest.json"), "patterns": groups, "files": len(files)}, ensure_ascii=False))
    return 0 if all(groups.values()) else 2
if __name__ == "__main__":
    raise SystemExit(main())
