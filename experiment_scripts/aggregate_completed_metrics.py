#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 complete.txt 中已完成场景的 metrics，导出为 CSV。

处理流程：
1) 对每个场景输出目录（默认 increment_50_150_noStaticMask）生成 metrics_summary.json
    - 仅提取各 JSON 内 key 以 _mean 结尾的字段
    - 若 metrics_summary.json 已存在，则跳过生成
2) 对所有场景的 metrics_summary.json 计算跨场景均值（aggregate row）并导出 CSV/XLS

默认输入:
  - complete 列表: experiment_scripts/complete.txt
  - 输出根目录:   output/mini_subset_exp
默认输出:
  - output/mini_subset_exp/completed_metrics_summary.csv

用法:
  python experiment_scripts/aggregate_completed_metrics.py
  python experiment_scripts/aggregate_completed_metrics.py \
      --complete-file experiment_scripts/complete.txt \
      --output-base output/mini_subset_exp \
    --metrics-subdir increment_50_150_noStaticMask \
      --csv-name completed_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总已完成场景的 metrics 到 CSV")
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser.add_argument(
        "--complete-file",
        type=Path,
        default=script_dir / "complete.txt",
        help="complete.txt 路径",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=project_dir / "output" / "mini_subset_exp",
        help="训练输出根目录（包含 SinglePhysics/DoublePhysics/TriplePhysics）",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="completed_metrics_summary.csv",
        help="输出 CSV 文件名",
    )
    parser.add_argument(
        "--metrics-subdir",
        type=str,
        default="increment_50_150_noStaticMask",
        help="每个场景下存放 metrics JSON 的子目录",
    )
    return parser.parse_args()


def flatten_metrics(data: Any, prefix: str = "") -> Dict[str, Any]:
    """
    扁平化 metrics_summary.json。
    - 跳过 per_view（通常非常大且 key 不统一）
    - 保留其余标量字段（含 trainingview_test_mean / novelview_test_mean / overall_mean 等）
    """
    flattened: Dict[str, Any] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "per_view":
                continue
            new_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten_metrics(value, new_prefix))
    elif isinstance(data, list):
        flattened[prefix] = json.dumps(data, ensure_ascii=False)
    else:
        flattened[prefix] = data

    return flattened


def build_metrics_dir(scene_path: str, output_base: Path, metrics_subdir: str) -> Path:
    scene = Path(scene_path.strip())
    scene_name = scene.name
    parent_name = scene.parent.name
    return output_base / parent_name / scene_name / metrics_subdir


def extract_mean_fields(data: Any, prefix: str = "") -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if key.endswith("_mean") and not isinstance(value, (dict, list)):
                extracted[new_prefix] = value
            if isinstance(value, dict):
                extracted.update(extract_mean_fields(value, new_prefix))
    return extracted


def build_scene_summary(metrics_dir: Path) -> tuple[Path, bool]:
    summary_path = metrics_dir / "metrics_summary.json"
    if summary_path.exists():
        return summary_path, False

    summary: Dict[str, Any] = {}
    if not metrics_dir.exists():
        return summary_path, False

    json_files = sorted(metrics_dir.glob("*.json"))
    for json_file in json_files:
        if json_file.name == "metrics_summary.json":
            continue
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            mean_fields = extract_mean_fields(data)
            for key, value in mean_fields.items():
                summary[f"{json_file.stem}.{key}"] = value
        except Exception:
            continue

    if summary:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        return summary_path, True

    return summary_path, False


def to_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def build_aggregate_row(rows: List[Dict[str, Any]], fieldnames: List[str]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    aggregate_row: Dict[str, Any] = {
        "scene_path": "__AGGREGATE_MEAN__",
        "scene_group": "ALL",
        "scene_name": "ALL",
        "metrics_file": "",
        "status": f"aggregate_mean_ok_rows={len(ok_rows)}",
    }

    for key in fieldnames:
        if key in aggregate_row:
            continue
        values = []
        for row in ok_rows:
            numeric = to_finite_float(row.get(key))
            if numeric is not None:
                values.append(numeric)
        if values:
            aggregate_row[key] = sum(values) / len(values)

    return aggregate_row


def write_tsv_as_xls(xls_path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    # Excel 可直接打开：UTF-8 BOM + TAB 分隔 + .xls 扩展名
    with xls_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    complete_file = args.complete_file.resolve()
    output_base = args.output_base.resolve()
    csv_path = output_base / args.csv_name
    xls_path = output_base / f"{Path(args.csv_name).stem}.xls"

    if not complete_file.exists():
        raise FileNotFoundError(f"complete file not found: {complete_file}")

    output_base.mkdir(parents=True, exist_ok=True)

    with complete_file.open("r", encoding="utf-8") as f:
        scenes = [line.strip() for line in f if line.strip()]

    if not scenes:
        print(f"[WARN] complete file is empty: {complete_file}")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["scene_path", "scene_group", "scene_name", "metrics_file", "status"])
            writer.writeheader()
        print(f"Empty CSV created: {csv_path}")
        return

    rows: List[Dict[str, Any]] = []
    total = len(scenes)
    found = 0
    generated_summary = 0

    for scene_path in scenes:
        metrics_dir = build_metrics_dir(scene_path, output_base, args.metrics_subdir)
        metrics_summary_file, created = build_scene_summary(metrics_dir)
        if created:
            generated_summary += 1

        row: Dict[str, Any] = {
            "scene_path": scene_path,
            "scene_group": Path(scene_path).parent.name,
            "scene_name": Path(scene_path).name,
            "metrics_file": str(metrics_summary_file),
        }

        if not metrics_summary_file.exists():
            row["status"] = "missing_metrics"
            rows.append(row)
            continue

        try:
            with metrics_summary_file.open("r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            metric_flat = flatten_metrics(metrics_data)
            row.update(metric_flat)
            row["status"] = "ok"
            found += 1
        except Exception as exc:
            row["status"] = f"parse_error: {exc}"

        rows.append(row)

    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())

    preferred_prefix = ["scene_path", "scene_group", "scene_name", "metrics_file", "status"]
    other_columns = sorted([col for col in all_columns if col not in preferred_prefix])
    fieldnames = preferred_prefix + other_columns

    aggregate_row = build_aggregate_row(rows, fieldnames)
    rows_with_aggregate = [aggregate_row] + rows

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_with_aggregate)

    write_tsv_as_xls(xls_path, fieldnames, rows_with_aggregate)

    print("=== Aggregation done ===")
    print(f"Complete scenes:        {total}")
    print(f"metrics_summary created:{generated_summary}")
    print(f"Metrics found & parsed: {found}")
    print(f"Metrics missing/error:  {total - found}")
    print(f"Aggregate row:          added")
    print(f"CSV saved to:           {csv_path}")
    print(f"XLS saved to:           {xls_path}")


if __name__ == "__main__":
    main()
