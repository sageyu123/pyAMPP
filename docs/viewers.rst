Model Viewers
=============

gxbox-view
----------

``gxbox-view`` opens an existing model HDF5 file in the 3D viewer without recomputing.

Expected inputs:

- A model file containing at least one of:
  - ``corona`` (preferred),
  - ``nlfff`` / ``pot`` (accepted and normalized), or
  - ``chromo`` with sufficient magnetic cube fields.

Usage:

.. code-block:: bash

   gxbox-view /path/to/model.h5

Optional file picker mode:

.. code-block:: bash

   gxbox-view --pick

gxrefmap-view
-------------

``gxrefmap-view`` is a 2D map browser for base maps and refmaps stored in model HDF5 files.

Expected layout:

- Base maps:
  - ``base/bx``, ``base/by``, ``base/bz``, ``base/ic``, ``base/chromo_mask``
  - base header in ``base/index`` (fallbacks: ``base/index_header``, ``base/wcs_header``)
- Refmaps:
  - ``refmaps/<map_id>/data``
  - ``refmaps/<map_id>/wcs_header``

Usage:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5

List available maps only:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5 --list

Start from a specific map:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5 --start AIA_94
