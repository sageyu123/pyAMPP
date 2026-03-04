pyAMPP HDF5 Model Format
========================

This page documents the current stage-file contract written by ``gx-fov2box`` and consumed by:

- ``gxbox-view`` (3D viewer),
- ``gxrefmap-view`` (base/refmap browser),
- resume/rebuild workflows via ``--entry-box``.

File Naming
-----------

Typical output naming pattern:

- ``hmi.M_720s.YYYYMMDD_HHMMSS.<region>.CEA.NONE.h5``
- same stem with stage suffixes ``POT``, ``BND``, ``NAS``, ``NAS.GEN``, ``NAS.CHR``

Common Groups (All Stages)
--------------------------

``base``
~~~~~~~~

2D boundary maps:

- ``base/bx``, ``base/by``, ``base/bz``
- ``base/ic``
- ``base/chromo_mask``
- ``base/index`` (IDL-compatible serialized header payload)

``metadata``
~~~~~~~~~~~~

Core provenance:

- ``metadata/id``: model identifier
- ``metadata/execute``: full command string used to create the model
- ``metadata/projection``: ``CEA`` or ``TOP``
- ``metadata/disambiguation``: e.g. ``HMI`` or ``SFQ``
- ``metadata/axis_order_2d`` and ``metadata/axis_order_3d`` when present
- ``metadata/vector_layout`` when present

``refmaps``
~~~~~~~~~~~

Optional context maps, each in its own subgroup:

- ``refmaps/<map_id>/data``
- ``refmaps/<map_id>/wcs_header``

``grid``
~~~~~~~~

Grid geometry and identity:

- ``grid/dx``, ``grid/dy``, ``grid/dz``
- ``grid/voxel_id`` (uint32)

Stage-Specific Groups
---------------------

NONE
~~~~

- Minimal model shell for downstream continuation.
- No solved extrapolation required.
- ``corona`` may exist as placeholder fields with ``attrs/model_type = "none"``.

POT
~~~

Potential-field stage in ``corona``:

- ``corona/bx``, ``corona/by``, ``corona/bz``
- ``corona/dr``
- ``corona/attrs/model_type = "pot"``
- ``corona/corona_base`` when available

BND
~~~

Boundary-conditioned stage in ``corona``:

- ``corona/bx``, ``corona/by``, ``corona/bz``
- ``corona/dr``
- ``corona/corona_base`` when available

NAS
~~~

NLFFF stage in ``corona``:

- ``corona/bx``, ``corona/by``, ``corona/bz``
- ``corona/dr``
- ``corona/attrs/model_type = "nlfff"`` (or equivalent)

NAS.GEN
~~~~~~~

Adds line-tracing products (stored in ``lines`` and/or stage-compatible chromo metadata):

- line indexing arrays (start/end/apex/seed)
- status/code arrays
- aggregate line properties (average field, physical length)

NAS.CHR
~~~~~~~

Adds chromospheric payload in ``chromo`` and finalized grid metadata:

- chromospheric thermodynamic variables (where generated)
- chromosphere-resolved field cubes (when present)
- final ``grid/voxel_id`` and geometry values used by viewers/renderers

Reference-map IDs
-----------------

Typical AIA-derived IDs used by viewers:

- ``AIA_94``, ``AIA_131``, ``AIA_171``, ``AIA_193``, ``AIA_211``, ``AIA_304``, ``AIA_335``
- ``AIA_1600``, ``AIA_1700`` (if UV requested)

Axis and Layout Notes
---------------------

- 2D base/refmap arrays are map-shaped with explicit WCS headers retained.
- 3D field arrays are stored in the package-native HDF5 convention used by current writers/readers.
- Viewers and adapters handle any required transpose/reindex for rendering and legacy compatibility.

Resume/Entry Compatibility
--------------------------

For robust resume behavior, entry files should include:

- ``metadata/execute`` (for reproducible command provenance and path recovery),
- valid ``metadata/id`` stage naming,
- required stage groups for the selected jump/rebuild mode.

When ``--entry-box`` is supplied, ``gx-fov2box`` detects stage availability and enforces required groups for continuation/rebuild paths.
