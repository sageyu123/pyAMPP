pyAMPP GUI Workflow
===================

This page documents the current ``pyampp`` command-builder GUI behavior and how it maps to ``gx-fov2box``.

Purpose
-------

The GUI is a thin command constructor for reproducible CLI runs. It:

- captures model parameters,
- resolves entry-box resume/rebuild behavior,
- displays the exact command before execution,
- streams command output in the status panel.

Data Repositories
-----------------

The GUI has three path fields:

- ``SDO Data Dir`` (mapped to ``--data-dir``)
- ``GX Models Dir`` (mapped to ``--gxmodel-dir``)
- ``Entry Box`` (mapped to ``--entry-box`` when set)

Current behavior:

- ``data-dir`` and ``gxmodel-dir`` persist across sessions via ``QSettings``.
- If an entry box contains a valid ``metadata/execute`` path and that path exists on the current machine, the GUI pre-fills the corresponding directory field.
- If execute paths are invalid on this machine, the GUI falls back to default local directories and warns the user.

Model Configuration
-------------------

Core inputs:

- observation time (UTC)
- center coordinates with one selected frame:
  - ``--hpc`` (helioprojective)
  - ``--hgc`` (Carrington)
  - ``--hgs`` (Stonyhurst)
- projection:
  - ``--cea`` (default)
  - ``--top``
- box dimensions:
  - ``--box-dims NX NY NZ``
- spatial scale:
  - ``--dx-km``
- padding:
  - ``--pad-frac`` (GUI percent value is converted to fraction)
- disambiguation:
  - ``HMI`` (default, no extra flag)
  - ``SFQ`` (adds ``--sfq`` when applicable)

Entry-Box Modes
---------------

When ``Entry Box`` is provided, the GUI detects the stage/type and offers three execution modes:

- ``Continue``:
  - continue from detected entry stage,
  - command is kept minimal (entry + paths + workflow flags).
- ``Rebuild from NONE``:
  - keep entry as source metadata/context but restart stage computation from NONE (``--rebuild-from-none``).
- ``Rebuild from OBS``:
  - recompute from observed maps and current GUI model parameters (``--rebuild``).

Pipeline Options and Stage Stops
--------------------------------

Save toggles:

- ``--save-empty-box``
- ``--save-potential``
- ``--save-bounds``
- ``--save-nas``
- ``--save-gen``

Stop rules:

- ``Stop after download`` -> ``--stop-after dl``
- ``NONE only`` -> ``--stop-after none``
- ``POT only`` -> ``--stop-after pot``
- ``NAS only`` -> ``--stop-after nas``
- ``GEN only`` (or CHR disabled) -> ``--stop-after gen``

Other stage controls:

- ``Skip NLFFF`` -> ``--use-potential``
- ``Skip line computation`` -> ``--skip-lines``

Context-map toggles:

- EUV checkbox -> ``--euv``
- UV checkbox -> ``--uv``

Command Display and Execution
-----------------------------

- The command preview updates live as inputs change.
- ``Run`` launches ``gx-fov2box`` as a subprocess.
- stdout/stderr are streamed into the status pane.
- A successful run reports generated model paths and stage timings in the log.

Operational Notes
-----------------

- If you modify package code, reinstall with ``pip install -e .`` before re-launching the GUI.
- Cached file reuse depends on ``--data-dir`` and timestamp/series compatibility with downloader filename matching.
