# PR 0.2.0 Checklist

## Scope

- [ ] Version bump to `0.2.0` reflected in packaging files.
- [ ] Downloader behavior restored and cache-compatible with current parity workflow.
- [ ] GUI path persistence validated (`data-dir` + `gxmodel-dir`).
- [ ] Docs updated for HDF5 format and GUI workflow.

## Files to Highlight in PR

- `setup.py`
- `pyproject.toml`
- `pyampp/version.py`
- `pyampp/_version.py`
- `docs/model_hdf5_format.rst`
- `docs/gui_workflow.rst`
- `docs/viewers.rst`
- `docs/index.rst`
- `docs/changelog.rst`
- `CHANGELOG.rst`
- `README.rst`

## Validation Notes

- Reinstall editable package after changes:
  - `pip install -e .`
- Smoke test:
  - `pyampp` (GUI launches, path persistence works)
  - `gx-fov2box --help`
  - Resume/rebuild from existing entry box with cached `jsoc_cache` data
- Documentation sanity:
  - `make -C docs html` (or equivalent Sphinx build)

## Suggested PR Title

`Release prep v0.2.0: downloader/cache compatibility, GUI persistence, and docs refresh`

## Suggested PR Summary (Short)

This PR prepares pyAMPP `v0.2.0` by updating package versioning, documenting current HDF5/GUI behavior, and stabilizing user workflows for iterative model production and resume/rebuild execution.
