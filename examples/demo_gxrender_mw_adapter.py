from __future__ import annotations

import argparse
from pathlib import Path

from pychmp import GXRenderMWAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo gxrender-backed map synthesis via pyCHMP adapter.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--ebtel-path", default=None)
    p.add_argument("--frequency-ghz", type=float, default=5.8)
    p.add_argument("--pixel-scale-arcsec", type=float, default=2.0)
    p.add_argument("--q0", type=float, default=0.0217)
    p.add_argument("--tbase", type=float, default=1e6)
    p.add_argument("--nbase", type=float, default=1e8)
    p.add_argument("--a", type=float, default=0.3)
    p.add_argument("--b", type=float, default=2.7)
    p.add_argument("--output-h5", default=None, help="Optional path to save rendered map as HDF5 file")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Parse output path if provided.
    output_dir = None
    output_name = None
    actual_output_file = None

    if args.output_h5:
        output_path = Path(args.output_h5)
        output_dir = str(output_path.parent)
        # Remove .h5 extension if provided (gxrender will add it after save).
        if output_path.suffix == ".h5":
            output_name = output_path.stem
            actual_output_file = output_path
        else:
            output_name = output_path.name
            actual_output_file = output_path.with_suffix(".h5")

    adapter = GXRenderMWAdapter(
        model_path=args.model_path,
        ebtel_path=args.ebtel_path,
        frequency_ghz=args.frequency_ghz,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        tbase=args.tbase,
        nbase=args.nbase,
        a=args.a,
        b=args.b,
        output_dir=output_dir,
        output_name=output_name,
        output_format="h5",
    )
    image = adapter.render(args.q0)

    print(f"Rendered map shape: {image.shape}")
    print(f"Rendered map min/max: {image.min():.4e}/{image.max():.4e}")

    if args.output_h5 and actual_output_file:
        print(f"\nMap saved to: {actual_output_file}")
        print(f"View with: gxrender-map-view {actual_output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
