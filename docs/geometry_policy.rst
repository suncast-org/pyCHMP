Observation Geometry Policy
===========================

pyCHMP observational workflows must infer the line of sight (LOS) from the
observation and pass that observer identity to pyGXrender. pyGXrender owns the
authoritative decision about whether a model-saved FOV can be reused or whether
an observer-dependent auto/inscribing FOV must be computed.

The policy is:

* External microwave observations are Earth LOS. Current radio maps are treated
  as Earth-view maps because pyCHMP does not support non-Earth radio-imaging
  spacecraft observations.
* AIA/SDO observations and internal AIA refmaps are Earth LOS.
* Other observations use the explicit instrument identity when it maps to a
  known observer. If the instrument is unknown, pyCHMP falls back to observer
  metadata in the map header or refmap metadata.
* pyCHMP does not implement the saved-FOV-vs-auto-FOV policy. It passes the
  inferred observer identity to the pyGXrender workflow policy and uses the
  geometry resolved by that upstream layer.
* Explicit user geometry or observer overrides remain explicit. They are logged
  and recorded as such, but the observation LOS is still recorded in artifact
  diagnostics.

This keeps pyCHMP's responsibility narrow: pyCHMP normalizes observation
identity, passes it consistently to the renderer, and records the upstream
geometry decision. Projection details and FOV policy remain the responsibility
of the pyCHMP -> pyGXrender -> pyAMPP chain.

The shared implementation is ``pychmp.geometry_policy``. The real-data
workflows record the selected mode and reason in artifact diagnostics using
``geometry_policy_*`` keys.
