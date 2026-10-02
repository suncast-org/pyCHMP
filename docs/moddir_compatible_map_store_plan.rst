modDir-Compatible Map Store Plan
================================

Purpose
-------

This document records the design and implementation plan for making the
pyCHMP map store behave like the proven GX Simulator ``modDir`` workflow.
The immediate motivation is the AR11520 EUV adaptive-search work, where a
171 A search appeared able to score incompatible maps that were produced while
working on 94 A. That must not be possible.

The redesign goal is not a narrow guard against one bad artifact. The goal is
to make rendered maps reusable across searches only when their physical and
render provenance is compatible, and to make channel selection happen at
scoring time in the same spirit as the IDL CHMP code.


IDL Reference Behavior
----------------------

The IDL implementation provides the reference semantics.

``gx_search4bestq.pro`` renders trial maps into a model directory
(``modDir``). Files are named by the EBTEL search coordinates, primarily
``a``, ``b``, and ``q0``. The directory is reusable across searches; it is not
owned by one search branch.

``gx_euvrender_ebtel.pro`` attaches the EBTEL provenance to the model through
``NTkey`` and writes that provenance into the map object's ``gx_key``. The
saved object is therefore not just an image array; it is a rendered model
product with internal provenance.

``gx_processmodels_ebtel.pro`` restores compatible map objects from
``modDir`` and selects the requested frequency or channel from the map object
at scoring time. For EUV image search, it compares the requested ``CHAN`` layer
to the corresponding observation. A 171 A search should therefore select the
171 A layer from a compatible model product; it should never interpret a 94 A
array as a 171 A map.

``gxModel::ComputeTRmask`` with ``type='Bz'`` uses ``abs(Bz) >=
abs(threshold)``. The pyCHMP option corresponding to that behavior is an
unsigned Bz threshold, for example ``|Bz| >= 200 G``.


Current Python Problem
----------------------

The current pyCHMP map store is too closely coupled to search/trial records.
It can persist individual arrays under ``map_store`` and link them from trial
rows, but the stored arrays do not always carry enough map-attached identity to
prove:

- which spectral channel or frequency they represent,
- which multi-channel render product they came from,
- which model/FOV/observer/response/EBTEL settings created them,
- whether they are raw forward-model maps or search-specific comparison
  products,
- whether a later search can safely reuse them without rerendering.

This makes the implementation both less reusable and less safe than IDL
``modDir``. A search should not need to trust a previous search branch to know
what a map is. The map store itself must carry that information.


Design Principles
-----------------

1. Treat ``map_store`` as a render-product repository.

   A stored render product belongs to a physical model/render identity, not to
   a search branch. Searches consume render products; they do not define their
   identity.

2. Store provenance with each reusable map/layer.

   Every reusable map or layer must carry a structured key equivalent in role
   to IDL ``gx_key``. A map may be moved out of its original repository and
   still describe its provenance. The repository may keep indices for fast
   lookup, but those indices are not the source of truth. The key should
   include model identity, geometry/FOV, observer policy, renderer
   configuration, EBTEL table, EBTEL formula, response table, ``a``, ``b``,
   ``q0``, and the spectral descriptor.

3. Keep multi-channel outputs grouped.

   When a render produces all AIA channels, the store should keep them as one
   product with a channel inventory. A later 94 A or 171 A search should select
   the needed layer from that product.

4. Separate render products from scoring products.

   Raw rendered evidence should be cached independently from metric masks,
   smoothing, PSF convolution, residual products, and metric values. PSF
   kernels, residual maps, metric masks, thresholds, shifts, and scores are
   search-specific state. They belong to the search identity or search results,
   not to the multi-search map store.

5. Treat PSFs as reference/scoring assets.

   PSFs are not synthetic render products. In the IDL workflow the PSF belongs
   to the reference map side and is retrieved when metrics are computed. pyCHMP
   should follow that semantic model: default channel/frequency PSFs may be
   cached in the artifact with the observation/reference payload, but the
   selected PSF and any user override are part of the search recipe. Changing a
   PSF changes the metric identity but must not change or contaminate reusable
   synthetic render products.

6. Preserve EUV decomposition when available.

   EUV transition-region and coronal components should be stored separately
   when the renderer can provide them. A changed TR mask or comparison mask can
   then recombine and rescore without rerendering.

7. Fail closed on ambiguous legacy entries.

   Until a map-store entry has sufficient identity, EUV/UV searches should not
   reuse it across channels. Ambiguous entries can remain in old artifacts but
   must not seed new scores.


Target Data Model
-----------------

The target artifact has separate synthetic-render and reference/scoring layers.

Render Product
~~~~~~~~~~~~~~

A render product is the pyCHMP equivalent of an IDL ``modDir`` map object.
It is identified by a stable render key and stores:

- ``render_product_id``
- structured ``gx_key``-like provenance
- model identity and model-file fingerprint
- FOV, pixel scale, canvas, observer, and LOS geometry
- EBTEL table path and fingerprint
- EBTEL ``q0`` and heating formula parameters
- renderer name/version and options
- response-table identity for EUV/UV
- channel/frequency inventory
- component inventory

Spectral Layers
~~~~~~~~~~~~~~~

Each product can contain multiple spectral layers:

- AIA 94, 131, 171, 193, 211, 304, 335 A
- microwave frequencies
- future UV/EUV channels

Each layer has an explicit spectral descriptor and carries its own
``gx_key``-like provenance. It is illegal for scoring code to infer the channel
from a previous search id or from a repository index alone.

Component Layers
~~~~~~~~~~~~~~~~

Each spectral layer may contain one or more component arrays:

- raw total model map
- coronal contribution
- transition-region contribution

Reusable component arrays are stored before comparison masks, PSF convolution,
normalization choices, or residual calculations are applied.

Convolved maps, residual maps, metric maps, and diagnostic comparison products
are not component layers of the reusable map store. They may be generated,
displayed, or cached under a search-specific results area, but they must not be
used as reusable render evidence.

Reference and PSF Assets
~~~~~~~~~~~~~~~~~~~~~~~~

Observed maps, uncertainty maps, WCS/geometry descriptors, and default
channel/frequency PSF kernels belong to the reference side of the artifact, not
to the synthetic render-product store. This follows the IDL CHMP design, where
the PSF is a property of the reference maps used during metric computation.

The artifact may cache PSF kernels so searches can reproduce convolved maps,
metrics, and residual panels without regenerating instrument responses. A search
uses the reference/default PSF unless its recipe explicitly selects a corrected
or replacement PSF. The selected PSF identity and fingerprint are part of the
search identity; the PSF kernel itself is not part of the reusable synthetic
map provenance.


Search-Time Behavior
--------------------

When a search evaluates ``(a, b, q0)``:

1. Build the required render-product key from the model, FOV, observer,
   renderer, EBTEL, response, and ``(a, b, q0)``.

2. Ask ``map_store`` for a compatible render product.

3. If a product exists, select the requested channel/frequency layer by its
   spectral descriptor.

4. If the needed layer exists, build the scoring map from stored components.
   For EUV, recombine TR/corona components using the current TR mask policy if
   component storage is available.

5. Apply the current scoring recipe: selected PSF, comparison mask, sigma
   policy, metric, and any fixed shifts. The selected PSF is resolved from the
   reference/default PSF assets or from an explicit search override.

6. Persist the trial score and link it to the render product and selected
   layer/component identities. Persist search-specific outputs, such as
   residuals or convolved maps, only under the search result namespace.

7. If no compatible product exists, render a new product, store all requested
   layers/components, and then score the selected layer.


Implementation Phases
---------------------

Phase 1: Document and Guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Write this plan and keep it on a clean branch.
- Add tests that reproduce the AR11520 failure pattern:

  - a 94 search populates the store,
  - a 171 search starts from the same artifact,
  - 171 scoring must not consume a 94-only map,
  - valid 171 maps may be reused only when their identity says 171.

- Keep fail-closed behavior for ambiguous EUV/UV map-store entries.

Phase 2: Structured Render Identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Define a canonical render-product key builder.
- Add a structured provenance schema to new map-store products.
- Include model, geometry, EBTEL, response, renderer, spectral inventory, and
  ``(a, b, q0)``.
- Add identity comparison helpers with clear diagnostics when reuse is denied.

Phase 3: Product-Centric Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Store grouped render products instead of unrelated scalar arrays.
- Preserve all rendered channels under the same product when rendered together.
- Keep search trial rows as links to product/layer/component ids.
- Ensure each reusable layer is self-describing, with provenance attached to
  the layer metadata, not only to an external map-store index.
- Provide a compatibility shim so old artifacts can still be viewed, repaired,
  or rescored conservatively.

Phase 4: Channel-Aware Rescoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Make Q0 rescoring select the target layer from a render product.
- Ensure one product can support independent 94 A and 171 A searches without
  sharing trial metrics.
- Separate render reuse from score reuse. A previous rendered map can be reused
  while metrics are recomputed for a different target channel or scoring recipe.
- Resolve PSFs from reference/default assets or explicit search overrides; do
  not look for PSF kernels in synthetic render-product provenance.

Phase 5: EUV Component Reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Extend the renderer adapter to store TR and coronal component maps when
  available.
- Recombine components under the requested TR-mask policy.
- Confirm that changes to metric threshold or TR mask do not force a rerender
  when reusable components exist.

Phase 6: Migration and Repair Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add an artifact inspection command that reports map-store products, channel
  inventory, component inventory, and ambiguous legacy entries.
- Add a conservative migration/repair path for old artifacts:

  - leave legacy arrays in place,
  - add structured identity only when it can be proven,
  - mark uncertain entries as non-reusable.


Acceptance Tests
----------------

The redesign is not complete until these behaviors are covered by tests:

- A 94 A search followed by a 171 A search cannot cross-score 94 maps as 171.
- A 94 A search can reuse compatible 94 A render products from the store.
- A 171 A search can reuse compatible 171 A layers rendered during a previous
  all-channel render.
- Purging a search branch does not delete reusable render products.
- Changing a metric mask causes rescoring, not rerendering, when reusable raw
  products exist.
- Changing the PSF recipe causes reconvolution/rescoring, not rerendering, and
  does not alter the reusable map-store product.
- Default PSFs are stored with the reference/observation payload, and a
  search-level PSF override creates a distinct scoring identity.
- Changing the TR mask causes recombination/rescoring, not rerendering, when
  TR/corona components exist.
- Residual maps and convolved maps are stored only under search-specific
  results, never as reusable map-store evidence.
- Ambiguous legacy EUV/UV map-store entries fail closed.
- Viewer labels and diagnostics show product identity, selected layer, and
  scoring recipe separately.


Open Questions
--------------

- Does the current pyGXrender path expose EUV TR/corona components directly,
  or do we need an adapter-level enhancement?
- Should the product store live inside each artifact HDF5 file only, or should
  pyCHMP also support an external shared map-store file analogous to IDL
  ``modDir``?
- How much of the IDL ``gx_key`` string format should be preserved for
  interoperability, versus represented as structured JSON/HDF5 attributes with
  a derived human-readable key?
- What is the desired migration policy for AR11520 artifacts already produced
  during the debugging session?


Current implementation status (known gap)
-----------------------------------------

As of the ``design/moddir-compatible-map-store`` branch:

- Channel fail-closed for EUV/UV ambiguous identities is implemented and tested
  (Phase 1 guard against 94↔171 cross-scoring).
- Structured provenance fields (model / EBTEL / geometry / response / projection
  hashes and related identity JSON) are **written** onto new map-store products.
- Warm-start and persisted ``slice_index`` lookup still match primarily on
  domain/channel + ``(a, b, q0)``. Full provenance-checked “compatible render
  product” reuse (Phases 2–4) is **not** complete: two maps with the same
  channel and ``(a, b, q0)`` but different model or response can still collide
  (last register wins). Do not claim modDir parity until hash-aware lookup lands.


Immediate Next Step
-------------------

Before implementation, inspect the current pyCHMP map-store writer, reader,
warm-start, and Q0 scoring paths and map them onto this plan. The first code
change should be a failing regression test for channel contamination, followed
by the smallest identity guard that makes that test pass on the clean branch.
