"""Command-line entry point for pyCHMP."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .ab_scan_artifacts import backfill_artifact_diagnostics, load_run_history


def _iter_backfill_targets(target: Path, *, recursive: bool) -> list[Path]:
    if target.is_dir():
        iterator = target.rglob("*.h5") if recursive else target.glob("*.h5")
        return sorted(path for path in iterator if path.is_file())
    return [target]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pychmp",
        description=(
            "Python Coronal Heating Modeling Pipeline (scaffold). "
            "Use this command as the entry point for future fitting workflows."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    history_parser = subparsers.add_parser(
        "artifact-history",
        help="Print recorded command provenance for a scan artifact.",
    )
    history_parser.add_argument("artifact_h5", type=Path, help="Artifact H5 path")
    history_parser.add_argument("--slice-key", default=None, help="Optional artifact slice key")
    history_parser.add_argument("--latest", action="store_true", help="Print only the latest recorded entry")
    history_parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")

    backfill_parser = subparsers.add_parser(
        "artifact-backfill",
        help="Populate missing compatibility diagnostics in an existing artifact.",
    )
    backfill_parser.add_argument("artifact_h5", type=Path, help="Artifact H5 path")
    backfill_parser.add_argument("--slice-key", default=None, help="Optional artifact slice key")
    backfill_parser.add_argument("--dry-run", action="store_true", help="Report proposed updates without modifying the artifact")
    backfill_parser.add_argument("--recursive", action="store_true", help="When the target is a directory, include subdirectories")
    backfill_parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")
    return parser


def _print_history_text(entries: list[dict], *, latest_only: bool) -> int:
    if not entries:
        print("No run history recorded in this artifact.")
        return 0
    selected_entries = [entries[-1]] if latest_only else entries
    if latest_only:
        print("Latest run history entry")
        print("")
    else:
        print(f"Run history entries: {len(selected_entries)}")
        print("")
    for index, entry in enumerate(selected_entries, start=1):
        print(f"[{index}] timestamp_utc = {entry.get('timestamp_utc', '')}")
        print(f"    action = {entry.get('action', '')}")
        print(f"    target_metric = {entry.get('target_metric', '')}")
        print(f"    cwd = {entry.get('cwd', '')}")
        print(f"    python_executable = {entry.get('python_executable', '')}")
        wrapper_command = entry.get("wrapper_command")
        if wrapper_command:
            print(f"    wrapper_command = {wrapper_command}")
        print(f"    effective_python_command = {entry.get('effective_python_command', '')}")
        print(f"    artifact_path = {entry.get('artifact_path', '')}")
        print(f"    log_path = {entry.get('log_path', '')}")
        if index != len(selected_entries):
            print("")
    return 0


def _print_backfill_text(report: dict) -> int:
    print(f"Artifact: {report.get('artifact_path', '')}")
    print(f"Dry run: {'yes' if report.get('dry_run') else 'no'}")
    print(f"Slices inspected: {report.get('slice_count', 0)}")
    print(f"Slices updated: {report.get('updated_slice_count', 0)}")
    updated_fields = dict(report.get("updated_fields", {}))
    skipped_fields = dict(report.get("skipped_fields", {}))
    if updated_fields:
        print("Updated fields:")
        for key in sorted(updated_fields):
            print(f"  {key}: {updated_fields[key]}")
    else:
        print("Updated fields: none")
    if skipped_fields:
        print("Skipped fields:")
        for key in sorted(skipped_fields):
            print(f"  {key}: {skipped_fields[key]}")
    print("")
    for item in list(report.get("slices", [])):
        print(f"[{item.get('slice_key', '')}] updated={'yes' if item.get('updated') else 'no'}")
        for key, value in sorted(dict(item.get("updated_fields", {})).items()):
            print(f"    set {key} = {value}")
        for key, value in sorted(dict(item.get("skipped_fields", {})).items()):
            print(f"    skipped {key}: {value}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "artifact-history":
        entries = load_run_history(args.artifact_h5, slice_key=args.slice_key)
        selected_entries = [entries[-1]] if args.latest and entries else entries
        if args.json:
            print(json.dumps(selected_entries, indent=2, sort_keys=True))
            return 0
        return _print_history_text(entries, latest_only=bool(args.latest))
    if args.command == "artifact-backfill":
        reports = [
            backfill_artifact_diagnostics(
                path,
                slice_key=args.slice_key,
                dry_run=bool(args.dry_run),
            )
            for path in _iter_backfill_targets(args.artifact_h5, recursive=bool(args.recursive))
        ]
        if args.json:
            print(json.dumps(reports, indent=2, sort_keys=True))
            return 0
        for index, report in enumerate(reports, start=1):
            if index > 1:
                print("")
            _print_backfill_text(report)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
