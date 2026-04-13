"""Command-line entry point for pyCHMP."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .ab_scan_artifacts import load_run_history


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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
