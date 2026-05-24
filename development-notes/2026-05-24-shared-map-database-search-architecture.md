# 2026-05-24 Shared Map Database Search Architecture

## Position

Yes, this is the intended direction.

The artifact should treat synthetic maps as shared, reusable science products,
while search slices own only the trial history, metric computation, and final
solution selection. Parameters such as metric choice, masks, beams, thresholds,
or search strategy influence how metrics are computed, but they do not change
the identity of the stored synthetic map itself.

That separation is the key to removing redundant rendering and making adaptive
search practical at scale.

## Core Principle

All maps stored in a given artifact share the same WCS geometry, data shape,
and spatial resolution.

Therefore the artifact should use a shared map database indexed by a stable map
identity, not by the search slice that happened to request it.

The search slice should record:

- which map was used for each trial
- the metrics computed from that map
- the trial ordering and search progression
- the final solution for that slice

It should not duplicate the synthetic map payload unless that payload is the
canonical shared copy.

## Map Identity

Each computed map needs a deterministic id that is sufficient to locate it
unambiguously in the artifact.

Recommended identity components:

- slice family or observation family
- `(a, b)` point identity
- `q0` or other trial parameter identity when relevant
- frequency or channel identity
- optional rendering mode or geometry signature only when it changes the map

Important rule:

- metrics, masks, beam-selection policy, and similar search parameters are not
  part of the synthetic-map identity unless they alter the actual rendered map

If they only affect metric computation, they belong to the search slice, not to
the map store key.

## Artifact Layout

### Shared map database

Use a shared map section in the artifact, for example:

- `/map_store/...`

This section stores the actual synthetic map products once, keyed by the map
identity.

### Search slices

Each search slice stores:

- slice diagnostics and compatibility signature
- trial history
- references to shared maps used by each trial
- computed metrics and any derived scalar summaries
- final best point / solution selection

Search slices may be multiple and may reuse the same shared maps if they share
compatible WCS geometry and map identity.

## Trial Workflow

For each trial point in a search slice:

1. compute or load the synthetic map for the requested `(a, b, q0, frequency/channel)` identity
2. compute the trial metrics from that map immediately
3. append the metrics to the slice trial history
4. append the map to the shared map store if it is new
5. store a pointer from the trial history entry to the shared map record
6. continue to the next trial until the point is solved
7. advance to the next grid point and repeat

This means the search slice never needs to own a copy of the map payload just to
preserve its history.

## Multi-Search Reuse Rules

A new search slice should first check whether a requested map already exists in
the shared store.

If the map exists:

- reuse it
- compute metrics from it directly
- append only the new trial reference and metrics to the search slice

If the map does not exist:

- compute the missing map
- send it to the dispatcher for logging and persistence
- then use it for metric computation and slice history recording

This is the mechanism that allows multiple slices to share the same rendered
maps while still maintaining independent metrics and final solutions.

## Dispatcher Model

The dispatcher should be the only writer to the artifact.

Workers may run in parallel, but they should not write the HDF5 file directly.

Recommended pattern:

- workers compute synthetic maps and metrics
- workers send completed map payloads plus metrics to a dispatcher queue
- dispatcher performs all artifact writes in a single serialized stream
- dispatcher appends the map to the shared store if new
- dispatcher records a reference into the trial history

This avoids HDF5 multi-writer races and keeps the artifact consistent.

### Thread/process guidance

- parallel workers should have read access to existing shared maps when they
  need to check for reuse
- one writer process or thread should own the file handle
- the dispatch queue should be bounded so memory cannot grow without limit

If CPU rendering dominates, process-based workers are usually safer than plain
threads. Threads are acceptable only if the renderer releases the GIL and the
implementation has been verified under load.

## Concurrency Invariants

1. The artifact must be append-only for shared map records.
2. Shared map ids must be stable and deterministic.
3. Trial history entries must store references, not duplicate payloads.
4. Only one writer may append to the HDF5 artifact at a time.
5. Search slices may read from the shared map store concurrently, but they do
   not write the file directly.
6. A map written once should be reusable by any compatible search slice.

## Compatibility Rules

Maps may be shared across slices only when the underlying map identity is the
same and the geometry contract matches.

The following are search-slice parameters, not map identity by themselves:

- target metric
- mask threshold
- mask type
- beam selection policy
- q0 search bounds when they do not alter the rendered map
- boundary strategy

The following may contribute to map identity only when they change the rendered
synthetic output:

- WCS geometry
- spatial resolution
- frequency or channel
- actual beam convolution applied to the map
- observation-specific render mode

## Expected Behavior

When a slice decides that a given `(a, b, q0, frequency/channel)` is needed for
the next trial:

1. check the shared map store first
2. if found, reuse the map for metric computation
3. if absent, compute it once
4. send the completed map to the dispatcher
5. record a trial-history pointer to the shared map entry

That is the efficient model.

## Open Design Questions

1. Should the shared map key be purely content-derived, purely parameter-derived,
   or hybrid?
2. Should the trial history pointer refer to a map path, a numeric map id, or a
   compact reference record with both?
3. Should the dispatcher deduplicate exact payloads across slices by content
   hash, or only by parameter identity?
4. Should the writer flush after every completed point, or batch flush for very
   high-throughput runs?

## Implementation Phases

### Phase 1

- introduce the shared map store as the canonical payload location
- record trial-history references to shared maps
- keep slice-level metric computation and final solution logic unchanged

### Phase 2

- move parallel execution behind a dispatcher queue
- make the dispatcher the only artifact writer
- add bounded back-pressure for memory safety

### Phase 3

- simplify resume logic so slices reuse compatible shared maps before any new
  rendering occurs
- remove redundant trial payload duplication from slice histories

## Conclusion

This is the right architecture for pyCHMP.

It keeps one canonical copy of each synthetic map, lets multiple slices reuse
that map safely, and keeps metrics/search policy separate from the rendered
science product itself.