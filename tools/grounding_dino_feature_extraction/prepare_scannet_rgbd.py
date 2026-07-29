#!/usr/bin/env python3
"""Extract ScanNet RGB-D frames with the official ScanNet SensReader."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _is_complete(scene_dir: Path) -> bool:
    return (
        any((scene_dir / "color").glob("*"))
        and any((scene_dir / "depth").glob("*"))
        and any((scene_dir / "pose").glob("*"))
        and (scene_dir / "intrinsic" / "intrinsic_color.txt").is_file()
        and (scene_dir / "intrinsic" / "intrinsic_depth.txt").is_file()
    )


def _select_scenes(args: argparse.Namespace) -> list[str]:
    if args.scene:
        scenes = list(dict.fromkeys(args.scene))
    elif args.scene_list:
        scenes = [
            line.strip()
            for line in args.scene_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        scenes = sorted(
            path.name for path in args.scans_root.iterdir() if path.is_dir()
        )
    if args.limit is not None:
        scenes = scenes[: args.limit]
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract color, depth, pose, and intrinsic data from ScanNet .sens files."
    )
    parser.add_argument("--scannet-root", type=Path, required=True)
    parser.add_argument(
        "--scans-root",
        type=Path,
        help="Defaults to <scannet-root>/scans.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <scannet-root>/ExtractedRGBD_GAP1.",
    )
    parser.add_argument(
        "--sens-reader",
        type=Path,
        required=True,
        help="Path to the official ScanNet SensReader/python/reader.py.",
    )
    parser.add_argument("--scene", action="append")
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.scannet_root = args.scannet_root.expanduser().resolve()
    args.scans_root = (
        args.scans_root.expanduser().resolve()
        if args.scans_root
        else args.scannet_root / "scans"
    )
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else args.scannet_root / "ExtractedRGBD_GAP1"
    )
    args.sens_reader = args.sens_reader.expanduser().resolve()

    if not args.scans_root.is_dir():
        parser.error(f"Scan directory not found: {args.scans_root}")
    if not args.sens_reader.is_file():
        parser.error(f"SensReader not found: {args.sens_reader}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    scenes = _select_scenes(args)
    if not scenes:
        parser.error("No scenes selected.")
    args.output_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for index, scene_id in enumerate(scenes, start=1):
        sens_file = args.scans_root / scene_id / f"{scene_id}.sens"
        scene_output = args.output_root / scene_id
        if not sens_file.is_file():
            failures.append((scene_id, f"missing {sens_file}"))
            continue
        if not args.overwrite and _is_complete(scene_output):
            print(f"[{index}/{len(scenes)}] skip {scene_id}")
            continue
        print(f"[{index}/{len(scenes)}] extract {scene_id}")
        command = [
            sys.executable,
            str(args.sens_reader),
            "--filename",
            str(sens_file),
            "--output_path",
            str(scene_output),
            "--export_depth_images",
            "--export_color_images",
            "--export_poses",
            "--export_intrinsics",
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((scene_id, f"SensReader exited with {exc.returncode}"))

    if failures:
        print("Failed scenes:", file=sys.stderr)
        for scene_id, reason in failures:
            print(f"  {scene_id}: {reason}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
