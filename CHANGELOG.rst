Changelog
=========

0.2.0
-----

Release focus:

- downloader compatibility restoration and cache reuse reliability,
- GUI workflow hardening for iterative model-production sessions,
- updated documentation for HDF5 stage format and GUI functionality.

Highlights:

- Restored downloader behavior while preserving IDL-style date folder layout (``YYYY-MM-DD``).
- Improved cache matching for existing HMI/AIA products across filename variants, reducing unnecessary re-downloads.
- Fixed missing-HMI edge cases during resume/rebuild paths when files were already present in cache.
- Made GUI repository path persistence robust for both:
  - ``--data-dir``
  - ``--gxmodel-dir``
- Default local data cache path uses ``~/pyampp/jsoc_cache``.
- Added/updated documentation:
  - ``docs/model_hdf5_format.rst``
  - ``docs/gui_workflow.rst``
  - ``docs/viewers.rst``

Packaging/versioning:

- Bumped package version to ``0.2.0`` in packaging metadata.
