#!/usr/bin/env python3
"""
初始化脚本：列出所有场景并创建必要的列表文件
用法:
  python init_scene_list.py [output_dir]
  python init_scene_list.py [output_dir] --update [--dry-run]

参数:
  --update   增量更新模式：按输出结果回算 complete/incomplete
  --dry-run  仅打印将要写入的变更，不实际写文件
"""

import sys
import argparse
from pathlib import Path

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()

# 定义场景根目录
SCENE_ROOTS = [
    "/media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/PhysicsBenchmark/mini_subset/Val/SinglePhysics",
    "/media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/PhysicsBenchmark/mini_subset/Val/DoublePhysics",
    "/media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/PhysicsBenchmark/mini_subset/Val/TriplePhysics",
]

WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_SCAN_ROOT = WORKSPACE_ROOT / "output" / "mini_subset_exp"
INCREMENT_OUTPUT_DIR = "increment_50_150_noStaticMask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="初始化/更新场景列表文件")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(SCRIPT_DIR),
        help="列表文件输出目录（默认: 脚本目录）",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="增量更新模式，只追加新场景并按 metrics 回算 complete/incomplete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要变更的内容，不实际写入文件",
    )
    return parser.parse_args()


def read_scene_set(path: Path):
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def write_scene_list(path: Path, scenes, dry_run: bool = False):
    if dry_run:
        print(f"[DRY-RUN] Would write {len(scenes)} lines to: {path}")
        return
    with path.open("w", encoding="utf-8") as f:
        for scene in scenes:
            f.write(scene + "\n")


def append_scene_list(path: Path, scenes, dry_run: bool = False):
    if not scenes:
        return
    if dry_run:
        print(f"[DRY-RUN] Would append {len(scenes)} lines to: {path}")
        return
    with path.open("a", encoding="utf-8") as f:
        for scene in scenes:
            f.write(scene + "\n")


def scan_all_scenes():
    all_scenes = []
    for root in SCENE_ROOTS:
        root_path = Path(root)
        if not root_path.is_dir():
            print(f"[WARN] Directory not found: {root}")
            continue

        print(f"Scanning: {root}")
        subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        for scene_dir in subdirs:
            all_scenes.append(str(scene_dir))
            print(f"  Found: {scene_dir.name}")
    return all_scenes


def has_any_json_in_increment_dir(scene_path: str, output_root: Path) -> bool:
    scene = Path(scene_path)
    scene_group = scene.parent.name
    scene_name = scene.name
    increment_dir = output_root / scene_group / scene_name / INCREMENT_OUTPUT_DIR

    if not increment_dir.is_dir():
        return False

    for _ in increment_dir.rglob("*.json"):
        return True

    return False


def init_mode(all_scenes, all_scenes_txt: Path, incomplete_txt: Path, complete_txt: Path, processing_txt: Path, dry_run: bool = False):
    print(f"\n=== Total scenes found: {len(all_scenes)} ===")

    write_scene_list(all_scenes_txt, all_scenes, dry_run=dry_run)
    write_scene_list(incomplete_txt, all_scenes, dry_run=dry_run)
    write_scene_list(complete_txt, [], dry_run=dry_run)
    write_scene_list(processing_txt, [], dry_run=dry_run)

    print("")
    if dry_run:
        print("[DRY-RUN] Would create/update files:")
    else:
        print("Created files:")
    print(f"  All scenes:   {all_scenes_txt}")
    print(f"  Incomplete:   {incomplete_txt}")
    print(f"  Complete:     {complete_txt}")
    print(f"  Processing:   {processing_txt}")


def update_mode(all_scenes, all_scenes_txt: Path, incomplete_txt: Path, complete_txt: Path, processing_txt: Path, dry_run: bool = False):
    existing_scenes = read_scene_set(all_scenes_txt)
    current_scenes = set(all_scenes)
    new_scenes = sorted(current_scenes - existing_scenes)

    if new_scenes:
        print(f"\n=== Found {len(new_scenes)} new scenes ===")
        for scene in new_scenes:
            print(f"  New: {Path(scene).name}")
        append_scene_list(all_scenes_txt, new_scenes, dry_run=dry_run)
    else:
        print("\n[INFO] No new scenes found in data roots.")

    current_scene_list = sorted(current_scenes)
    complete_scenes = set()
    print(f"[SCAN] checking JSON under: {DEFAULT_OUTPUT_SCAN_ROOT}/<group>/<scene>/{INCREMENT_OUTPUT_DIR}")
    for scene in current_scene_list:
        if has_any_json_in_increment_dir(scene, DEFAULT_OUTPUT_SCAN_ROOT):
            complete_scenes.add(scene)

    write_scene_list(complete_txt, sorted(complete_scenes), dry_run=dry_run)
    write_scene_list(processing_txt, [], dry_run=dry_run)

    incomplete_scenes = sorted(current_scenes - complete_scenes)
    write_scene_list(incomplete_txt, incomplete_scenes, dry_run=dry_run)

    print("")
    if dry_run:
        print("[DRY-RUN] Would update files:")
    else:
        print("Updated files:")
    print(f"  All scenes:   {all_scenes_txt}")
    print(f"  Complete:     {complete_txt}")
    print(f"  Processing:   {processing_txt}")
    print(f"  Incomplete:   {incomplete_txt}")
    print(f"  Output scan:  {DEFAULT_OUTPUT_SCAN_ROOT}")
    print(f"  Complete rule: {INCREMENT_OUTPUT_DIR} exists and contains any *.json")
    print(f"\n✓ Total new scenes added to all_scenes: {len(new_scenes)}")
    print(f"✓ Total scenes marked complete by metrics: {len(complete_scenes)}")
    print(f"✓ Total scenes in processing after sync: 0")
    print(f"✓ Total scenes in incomplete after sync: {len(incomplete_scenes)}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    all_scenes_txt = output_dir / "all_scenes.txt"
    incomplete_txt = output_dir / "incomplete.txt"
    complete_txt = output_dir / "complete.txt"
    processing_txt = output_dir / "processing.txt"
    log_dir = output_dir / "logs"

    # 创建输出目录和日志目录
    if args.dry_run:
        print(f"[DRY-RUN] output_dir: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    print("=== Scanning scenes from root directories ===")
    all_scenes = scan_all_scenes()

    if args.update:
        update_mode(
            all_scenes,
            all_scenes_txt,
            incomplete_txt,
            complete_txt,
            processing_txt,
            dry_run=args.dry_run,
        )
    else:
        init_mode(
            all_scenes,
            all_scenes_txt,
            incomplete_txt,
            complete_txt,
            processing_txt,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print(f"  Logs dir:     {log_dir}")
    print("")
    print(f"Total scenes in database: {len(set(all_scenes))}")
    mode = "update" if args.update else "init"
    print(f"Mode: {mode} ({'dry-run' if args.dry_run else 'write'})")


if __name__ == "__main__":
    main()
