# pyCHMP Demo Scripts

This folder contains demonstration scripts that exercise pyCHMP APIs outside of pytest.

## Scripts

- `demo_synthetic_q0_recovery.py`
  - Uses a synthetic renderer and verifies that fitting recovers a planted Q0.
  - No gxrender dependency required.

- `demo_gxrender_mw_adapter.py`
  - Shows how to instantiate `GXRenderMWAdapter` for a real model path.
  - Renders a single brightness temperature map and optionally saves to HDF5 for visualization.
  - Requires gxrender to be installed and valid model/EBTEL inputs.

## Usage

Run from repository root:

```bash
python tests/scripts/demo_synthetic_q0_recovery.py
```

```bash
python tests/scripts/demo_gxrender_mw_adapter.py \
  --model-path /path/to/test.chr.h5 \
  --ebtel-path /path/to/ebtel.sav \
  --frequency-ghz 5.8 \
  --pixel-scale-arcsec 2.0 \
  --q0 0.0217
```

To save and visualize the rendered map:

```bash
python tests/scripts/demo_gxrender_mw_adapter.py \
  --model-path /path/to/test.chr.h5 \
  --ebtel-path /path/to/ebtel.sav \
  --frequency-ghz 5.8 \
  --pixel-scale-arcsec 2.0 \
  --q0 0.0217 \
  --output-h5 /tmp/rendered_map.h5

# View the result:
gxrender-map-view /tmp/rendered_map.h5
```
