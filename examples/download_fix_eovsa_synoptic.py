#!/usr/bin/env python
"""Download EOVSA synoptic FITS files, optionally fix time cards, and rename by frequency."""

from __future__ import annotations

import argparse
import fnmatch
import io
from dataclasses import dataclass
from datetime import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

from astropy.io import fits
from astropy.time import Time


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)
                return


@dataclass(frozen=True)
class SourceEntry:
    filename: str
    url: str | None = None
    path: Path | None = None


@dataclass(frozen=True)
class PlannedEntry:
    source: SourceEntry
    output_path: Path
    freq_ghz: float
    source_time: Time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="Public synoptic day URL or local directory containing synoptic FITS files",
    )
    parser.add_argument("output_dir", type=Path, help="Directory where corrected FITS files will be written")
    parser.add_argument(
        "--match-glob",
        default="eovsa.synoptic_daily*.tb.disk.fits",
        help="Filename glob applied to links discovered on the page (default: daily products only)",
    )
    parser.add_argument(
        "--target-time",
        default=None,
        help="Optional UTC time to enforce in DATE-OBS/DATE on the original calendar date; if omitted, preserve source times",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for normalized output names, e.g. fixed_",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads and output names only")
    return parser


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else f"{url}/"


def _is_http_url(source: str) -> bool:
    return urlparse(source).scheme in {"http", "https"}


def _download_text(url: str) -> str:
    with urlopen(url) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _download_bytes(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def _read_source_bytes(source: SourceEntry) -> bytes:
    if source.path is not None:
        return source.path.read_bytes()
    if source.url is not None:
        return _download_bytes(source.url)
    raise ValueError(f"Source entry has neither local path nor URL: {source.filename}")


def _find_url_entries(day_url: str, match_glob: str) -> list[SourceEntry]:
    html = _download_text(day_url)
    parser = LinkCollector()
    parser.feed(html)
    entries: dict[str, SourceEntry] = {}
    for href in parser.hrefs:
        parsed = urlparse(href)
        candidate = Path(parsed.path).name
        if not candidate:
            continue
        if not fnmatch.fnmatch(candidate, match_glob):
            continue
        url = urljoin(day_url, href)
        entries[candidate] = SourceEntry(filename=candidate, url=url)
    return sorted(entries.values(), key=lambda item: item.filename)


def _find_local_entries(source_dir: Path, match_glob: str) -> list[SourceEntry]:
    if not source_dir.is_dir():
        raise SystemExit(f"Local source directory does not exist: {source_dir}")
    entries: dict[str, SourceEntry] = {}
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if not fnmatch.fnmatch(path.name, match_glob):
            continue
        entries[path.name] = SourceEntry(filename=path.name, path=path)
    return sorted(entries.values(), key=lambda item: item.filename)


def _find_source_entries(source: str, match_glob: str) -> list[SourceEntry]:
    if _is_http_url(source):
        return _find_url_entries(_ensure_trailing_slash(source), match_glob)
    return _find_local_entries(Path(source).expanduser(), match_glob)


def _parse_target_time(text: str | None) -> time | None:
    if text is None:
        return None
    parts = text.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS for --target-time, got {text!r}")
    hour, minute, second = (int(part) for part in parts)
    return time(hour=hour, minute=minute, second=second)


def _looks_like_freq_axis(header: fits.Header) -> bool:
    ctype3 = str(header.get("CTYPE3", "")).strip().upper()
    return bool(header.get("CRVAL3")) and (not ctype3 or ctype3 == "FREQ")


def _pick_reference_hdu_index(hdul: fits.HDUList) -> int:
    for idx, hdu in enumerate(hdul):
        if _looks_like_freq_axis(hdu.header):
            return idx
    raise ValueError("No HDU with usable frequency metadata (CRVAL3/CTYPE3) was found")


def _extract_freq_ghz(header: fits.Header) -> float:
    if "CRVAL3" not in header:
        raise ValueError("Missing CRVAL3 frequency card")
    value = float(header["CRVAL3"])
    unit = str(header.get("CUNIT3", "Hz")).strip().lower()
    if unit in {"hz", ""}:
        return value / 1e9
    if unit == "khz":
        return value / 1e6
    if unit == "mhz":
        return value / 1e3
    if unit == "ghz":
        return value
    raise ValueError(f"Unsupported frequency unit in CUNIT3: {header.get('CUNIT3')!r}")


def _format_timestamp_like(original_text: str, target: Time) -> str:
    original_text = str(original_text)
    if "T" in original_text:
        return target.isot.split(".")[0]
    return target.strftime("%Y-%m-%d %H:%M:%S.000")


def _rewrite_time_cards(header: fits.Header, target: Time) -> None:
    for key in ("DATE-OBS", "DATE"):
        if key in header:
            header[key] = _format_timestamp_like(header[key], target)
    if "MJD-OBS" in header:
        header["MJD-OBS"] = float(target.mjd)


def _build_output_name(prefix: str, target: Time, freq_ghz: float) -> str:
    stamp = target.strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}eovsa.synoptic_daily.{stamp}.f{freq_ghz:.3f}GHz.tb.disk.fits"


def _normalized_target_time(source_header: fits.Header, target_clock: time) -> Time:
    source_text = source_header.get("DATE-OBS") or source_header.get("DATE")
    if not source_text:
        raise ValueError("Could not determine source observation date from DATE-OBS or DATE")
    source_time = Time(str(source_text), format="isot" if "T" in str(source_text) else "iso", scale="utc")
    target_text = (
        f"{source_time.to_datetime().date().isoformat()} "
        f"{target_clock.hour:02d}:{target_clock.minute:02d}:{target_clock.second:02d}"
    )
    return Time(target_text, format="iso", scale="utc")


def _extract_source_time(source_header: fits.Header) -> Time:
    source_text = source_header.get("DATE-OBS") or source_header.get("DATE")
    if not source_text:
        raise ValueError("Could not determine source observation date from DATE-OBS or DATE")
    return Time(str(source_text), format="isot" if "T" in str(source_text) else "iso", scale="utc")


def _plan_one(source: SourceEntry, output_dir: Path, target_clock: time | None, prefix: str) -> PlannedEntry:
    payload = _read_source_bytes(source)
    with fits.open(io.BytesIO(payload), mode="readonly") as hdul:
        ref_idx = _pick_reference_hdu_index(hdul)
        ref_header = hdul[ref_idx].header
        freq_ghz = _extract_freq_ghz(ref_header)
        source_time = _extract_source_time(ref_header)
        effective_time = _normalized_target_time(ref_header, target_clock) if target_clock is not None else source_time
        output_name = _build_output_name(prefix=prefix, target=effective_time, freq_ghz=freq_ghz)
        output_path = output_dir / output_name
    return PlannedEntry(source=source, output_path=output_path, freq_ghz=freq_ghz, source_time=source_time)


def _sort_plan_preference(item: PlannedEntry, target_clock: time | None) -> tuple[int, float, str]:
    is_daily = 0 if ".synoptic_daily." in item.source.filename else 1
    if target_clock is None:
        return (is_daily, 0.0, item.source.filename)
    target_seconds = target_clock.hour * 3600 + target_clock.minute * 60 + target_clock.second
    source_dt = item.source_time.to_datetime()
    source_seconds = source_dt.hour * 3600 + source_dt.minute * 60 + source_dt.second
    distance = abs(source_seconds - target_seconds)
    return (is_daily, distance, item.source.filename)


def _build_plan(entries: Iterable[SourceEntry], output_dir: Path, target_clock: time | None, prefix: str) -> tuple[list[PlannedEntry], dict[Path, list[PlannedEntry]]]:
    raw = [_plan_one(entry, output_dir, target_clock, prefix) for entry in entries]
    grouped: dict[Path, list[PlannedEntry]] = {}
    for item in raw:
        grouped.setdefault(item.output_path, []).append(item)
    selected: list[PlannedEntry] = []
    for output_path in sorted(grouped, key=lambda path: path.name):
        choices = sorted(grouped[output_path], key=lambda item: _sort_plan_preference(item, target_clock))
        selected.append(choices[0])
    return selected, grouped


def _process_one(plan: PlannedEntry, target_clock: time | None, overwrite: bool) -> Path:
    payload = _read_source_bytes(plan.source)
    with fits.open(io.BytesIO(payload), mode="readonly") as hdul:
        ref_idx = _pick_reference_hdu_index(hdul)
        ref_header = hdul[ref_idx].header
        output_path = plan.output_path
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        if target_clock is not None:
            target_time = _normalized_target_time(ref_header, target_clock)
            for hdu in hdul:
                _rewrite_time_cards(hdu.header, target_time)
        hdul.writeto(output_path, overwrite=overwrite)
        return output_path


def _render_plan(entries: Iterable[SourceEntry], output_dir: Path, target_clock: time | None, prefix: str) -> list[str]:
    lines: list[str] = []
    selected, grouped = _build_plan(entries, output_dir, target_clock, prefix)
    for item in selected:
        collisions = grouped[item.output_path]
        if len(collisions) == 1:
            lines.append(f"{item.source.filename} -> {item.output_path}")
            continue
        skipped = ", ".join(candidate.source.filename for candidate in collisions if candidate.source.filename != item.source.filename)
        lines.append(f"{item.source.filename} -> {item.output_path}  [selected over: {skipped}]")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    target_clock = _parse_target_time(args.target_time)
    entries = _find_source_entries(str(args.source), str(args.match_glob))
    if not entries:
        raise SystemExit(f"No inputs matching {args.match_glob!r} were found at {args.source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for line in _render_plan(entries, output_dir, target_clock, str(args.prefix)):
            print(line)
        return 0
    plans, _ = _build_plan(entries, output_dir, target_clock, str(args.prefix))
    written: list[Path] = []
    for plan in plans:
        out_path = _process_one(
            plan=plan,
            target_clock=target_clock,
            overwrite=bool(args.overwrite),
        )
        written.append(out_path)
        print(f"✓ Wrote {out_path.name}")
    print(f"Wrote {len(written)} corrected FITS files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
