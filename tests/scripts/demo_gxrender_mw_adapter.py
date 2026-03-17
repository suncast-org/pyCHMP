from __future__ import annotations

import argparse

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
    return p.parse_args()


def main() -> int:
    args = parse_args()

    adapter = GXRenderMWAdapter(
        model_path=args.model_path,
        ebtel_path=args.ebtel_path,
        frequency_ghz=args.frequency_ghz,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        tbase=args.tbase,
        nbase=args.nbase,
        a=args.a,
        b=args.b,
    )
    image = adapter.render(args.q0)

    print(f"Rendered map shape: {image.shape}")
    print(f"Rendered map min/max: {image.min():.4e}/{image.max():.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
