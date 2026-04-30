Artifact Data Contract
======================

Purpose
-------

This document defines the target artifact architecture for pyCHMP search and
fitting workflows. It is intentionally broader than the current implementation:
it records the agreed contract that future work should converge toward, then
marks the implementation slices that are already in place.

The core design decision is that pyCHMP artifacts describe a *search over trial
states*, not merely a final PNG panel or a single finalized map. Single-point,
fixed-grid, sparse-grid, and adaptive searches should therefore converge on the
same canonical artifact model, differing only in how many trial records and
search coordinates they contain.

Design Principles
-----------------

1. One canonical artifact family.

   Single-point MW fits, single-point EUV fits, rectangular `(a, b)` scans,
   sparse scans, and adaptive scans should all target the same conceptual
   schema.

2. Generic slice identity.

   Artifacts must not assume a microwave-only frequency axis. Slice identity is
   represented generically via:

   - ``spectral_domain``: ``mw``, ``euv``, ``uv``, or another future domain
   - ``spectral_label``: human-facing text such as ``2.874 GHz`` or ``171 A``
   - optional domain-specific numeric metadata such as ``frequency_ghz`` or
     ``wavelength_angstrom``

3. Persist evidence, not only derived views.

   The canonical payload should prioritize the rendered and selected evidence
   required to reproduce or recompute downstream products. Convolved maps,
   residual maps, and metrics are derived products and may be cached, but are
   not the fundamental unit of storage.

4. Separate forward-model decomposition from comparison masks.

   EUV transition-region masks are part of the forward-model decomposition
   layer. Metric masks are part of the comparison/evaluation layer. They must
   remain separate in both schema and code.

5. One viewer contract.

   ``pychmp-view`` should adapt to whatever subset of the canonical data is
   present. It should not require a different file type for single-point versus
   scan artifacts.


Canonical Architecture
----------------------

The target artifact has three main layers.

Common Layer
~~~~~~~~~~~~

The common layer stores information shared by every trial in a slice or search:

- artifact contract version
- model / observation / EBTEL provenance
- observer and geometry metadata
- shared observational target maps
- shared ``B_los`` reference map
- canonical slice descriptors
- the designated optimization target slice
- logging policy metadata
- run history / command provenance

Slice Descriptors
~~~~~~~~~~~~~~~~~

Every artifact should expose a canonical list of slice descriptors. A slice
descriptor is generic and not microwave-specific.

Required logical fields:

- ``key``: stable machine identifier
- ``domain``: ``mw``, ``euv``, ``uv``, or future value
- ``label``: compact human-facing label
- ``display_label``: viewer-facing label
- ``role``: ``target`` or ``auxiliary``
- ``is_target``: boolean convenience flag

Optional domain-specific fields:

- ``frequency_ghz``
- ``wavelength_angstrom``
- ``channel_label``

The designated optimization slice is stored separately as ``target_slice_key``.
This allows an artifact to log multiple rendered slices while optimizing only
one of them.

Trial Layer
~~~~~~~~~~~

The canonical trial record is the fundamental search unit. A trial record should
be able to capture:

- search coordinates (for example ``a``, ``b``) when applicable
- ``q0``
- success / status / timing / optimizer metadata
- raw rendered map cubes for logged slices
- optional EUV component cubes:

  - coronal component cube
  - transition-region component cube
  - EUV TR-region mask

- metric mask information:

  - explicit realized metric mask, when available
  - metric-mask generation rule / metadata

- optional cached derived products:

  - convolved map cubes
  - residual cubes
  - metric tables

Solution Layer
~~~~~~~~~~~~~~

The final solution should be a convenience view of the stored trial history,
not a separate artifact concept. Redundant convenience payloads are acceptable
for the selected final solution, for example:

- final raw map(s)
- final convolved map(s)
- final residual map(s)
- final metric summary
- panel-ready quick-look data


Logging Policy Versus Recomputation
-----------------------------------

The artifact contract distinguishes between:

- **required evidence** that should be logged
- **derived products** that should be recomputable from the logged evidence

For future reuse under changed assumptions, the minimal important evidence is:

- observational maps
- raw synthetic map cubes
- PSF specification
- metric-mask metadata and/or explicit metric mask
- EUV component cubes and TR-region mask when relevant

From that evidence, downstream code should be able to recompute:

- convolved maps
- residual maps
- metrics under the original assumptions
- metrics under changed masks or changed target metric
- alternative total EUV maps under changed TR-mask assumptions


Metric Masks
------------

Metric masks must support at least two categories:

1. rule-derived masks, such as threshold / contour-percentage masks
2. explicit user-supplied 2D bit masks

Artifacts should therefore support both:

- a stored realized mask for exact reproduction
- stored mask-generation metadata / rule inputs for recomputation


EUV-Specific Requirements
-------------------------

The EUV path must preserve the distinction between:

- coronal EUV component
- transition-region EUV component
- TR-region bit mask
- metric mask

The TR-region mask must be logged separately from metric masks. This allows a
future workflow to recompute total EUV comparison maps from already stored
coronal and TR component cubes under changed TR-mask assumptions, without
rerunning ``pyGXrender``.

Current TR-Mask Assumption
--------------------------

The current pyCHMP implementation assumes that the EUV TR mask acts as a
downstream gate on an already selected and already computed TR contribution,
not as an upstream input that changes TR voxel identity or coronal/TR physics.

This assumption is consistent with the current code paths inspected during the
initial EUV artifact design:

- the current Python ``pyGXrender`` EUV workflow returns separate
  ``flux_corona`` and ``flux_tr`` cubes and forms the total map by linear
  superposition
- the historical IDL ``gx_box2id.pro`` implementation selects one TR voxel per
  LOS first, then uses ``tr_mask`` only to remove the ``/euv`` activity bit
  from that already selected TR voxel

Under this assumption, pyCHMP may safely:

- store coronal and TR component cubes per trial
- store the 2-D TR mask used for that run
- recompute total EUV maps later as ``corona + masked_tr`` without rerendering

Current DLL Nuance
~~~~~~~~~~~~~~~~~~

The currently inspected Python / DLL path adds one more practical nuance:

- the RenderGRFF / ``pyGXrender`` EUV DLL interface currently returns separate
  ``flux_corona`` and ``flux_tr`` cubes
- the returned ``flux_tr`` corresponds to the full TR contribution
- the DLL result does not currently expose the input or recovered 2-D TR mask

This differs from the historical IDL wrapper layer, where the wrapper recovers
the TR mask from voxel IDs and exposes both:

- the full TR contribution
- the recovered TR mask

before forming the masked TR and total maps.

Therefore, for current Python-side parity with the IDL wrapper semantics,
pyCHMP must explicitly preserve the TR mask alongside the returned EUV
component cubes. The canonical artifact should not assume that the DLL already
returned a masked TR map or that the TR mask can be recovered from the Python
result object alone.

Change Point
~~~~~~~~~~~~

If future work shows that changing the TR mask must also change which voxel is
considered the emitting TR voxel, or otherwise alters the upstream EUV
rendering physics, then this assumption must be revised and the stored TR
component can no longer be treated as mask-agnostic.

The first places to inspect and update in that case are:

- IDL reference path: ``gx_simulator/util/gx_box2id.pro`` and
  ``gx_simulator/euv/suport/gx_euv.pro``
- Python parity path: ``gxrender/io/voxel_id.py`` and the EUV workflow / SDK
  entrypoints in ``gxrender/euv.py`` and ``gxrender/sdk.py``
- pyCHMP recombination and artifact logic: ``pychmp/gxrender_adapter.py`` and
  the canonical artifact writer / reader path


Initial Implementation Slices
-----------------------------

The full migration will happen in phases. The current implementation plan is:

Phase 1
~~~~~~~

- Add this tracked contract document.
- Persist canonical slice descriptors in artifact common metadata.
- Persist the designated ``target_slice_key`` in artifact common metadata.
- Persist trial-logging policy metadata in artifact common metadata.

Phase 2
~~~~~~~

- Add a canonical single-point artifact writer on top of the canonical scan
  artifact machinery.
- Stop treating the current single-point artifact layout as the primary writer.

Phase 3
~~~~~~~

- Refactor ``fit_q0_obs_map.py`` to emit the canonical viewer-compatible
  artifact by default.
- Keep PNG generation as a derived output, not the defining schema.

Phase 4
~~~~~~~

- Extend ``pychmp-view`` to open single-point artifacts naturally via the same
  canonical payload path as scan/adaptive artifacts.

Phase 5
~~~~~~~

- Add optional per-trial raw map cube logging and richer EUV component logging.
- Add optional cached derived products for convenience.


What Is Implemented Now
-----------------------

As of the first slice corresponding to this document:

- canonical slice descriptors are persisted in the artifact common group
- the designated target slice key is persisted in the artifact common group
- a canonical trial-logging policy record is persisted in the artifact common
  group

These additions establish the schema backbone without yet forcing the full
single-point/scan writer migration.

As of the current canonical single-point writer slice:

- single-point fits are written as canonical sparse/viewer-compatible artifacts
- per-trial scalar histories are persisted in the canonical point record
- optional per-trial raw modeled maps, convolved modeled maps, and residual
  maps are persisted when available from the one-point fitting workflow
