# Include spectral identity in point progress logs

**Resolved 2026-09-20.** Implemented: point-start messages now include the target spectral label in both worker and new-render callback paths. EUV labels include the instrument and wavelength; MW labels include GHz. Tests exercise real point-start output.

The investigation notes below describe the original behavior.

Requested while the AR11520 six-channel search was running on 2026-09-20.
Implement after the current run finishes; do not interrupt it for this change.

Every point-search start message should identify the target channel/wavelength
or frequency, including both new-render and warm-start paths. The current
message is ambiguous when following the shared live log:

```text
Starting point: a=-0.100 b=0.300 (rendering new trials)
```

Desired examples:

```text
Starting point: AIA 193 Å; a=-0.100 b=0.300 (rendering new trials)
Starting point: 17 GHz; a=-0.100 b=0.300 q0_range=(...)
```

Use the active search's target spectral identity, not the list of auxiliary
channels rendered or stored by the same trial. Include units and the instrument
when available. Cover every `Starting point:` variant so the shared log makes
the currently optimized wavelength or frequency clear without scrolling back
to the channel header.
