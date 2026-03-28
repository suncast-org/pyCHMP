# EOVSA Synoptic Download And Time Fix Utility

## Goal

Provide a small CLI utility that:

1. downloads public EOVSA synoptic FITS files from a given day URL
2. preserves the original FITS structure and beam metadata
3. rewrites observation-time cards to a user-selected UTC time on the same day
4. renames each file using its header frequency in GHz instead of the source band token

This is intended to avoid the earlier workflow where manual FITS rewriting accidentally dropped `BMAJ`, `BMIN`, and `BPA`.

## Source URL Pattern

The public source layout is assumed to be:

```text
https://ovsa.njit.edu/fits/synoptic/YYYY/MM/DD/
```

The downloader matches page links against:

```text
eovsa.synoptic*.tb.disk.fits
```

## Output Filename Policy

The normalized output filename is:

```text
{prefix}eovsa.synoptic_daily.{YYYYMMDD}T{HHMMSS}Z.f{FREQ_GHZ:.3f}GHz.tb.disk.fits
```

Examples:

```text
eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits
fixed_eovsa.synoptic_daily.20201126T200000Z.f4.332GHz.tb.disk.fits
```

Notes:

- `prefix` is optional and user-controlled
- frequency comes from FITS header `CRVAL3` with unit conversion from Hz to GHz
- the normalized stem intentionally removes the original source band token such as `s02-04`

## Header Rewrite Policy

The utility must preserve all HDUs and all unrelated header cards.

It only rewrites time-related cards when present:

- `DATE-OBS`
- `DATE`

The replacement timestamp uses the original calendar date from the source file and the user-supplied UTC time.

Beam and frequency cards must remain untouched:

- `BMAJ`
- `BMIN`
- `BPA`
- `CRVAL3`
- `CUNIT3`

## HDU Selection Rules

Many EOVSA synoptic FITS files store the image metadata in a compressed image extension rather than the primary HDU.

The utility therefore:

1. scans HDUs in order
2. picks the first HDU header containing usable frequency metadata:
   - `CRVAL3`
   - optionally `CTYPE3 == FREQ`
3. uses that header both for:
   - frequency-based renaming
   - time-card updates

If the primary HDU also has matching time cards, those may be updated too, but the script must not require that.

## Safety

- do not overwrite existing outputs unless `--overwrite` is supplied
- skip source links that do not parse as the target FITS naming pattern
- fail clearly if a downloaded FITS file lacks usable frequency metadata
- preserve the downloaded original bytes only in memory; write only the corrected output file
