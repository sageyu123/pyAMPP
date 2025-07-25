pyAMPP: Python Automatic Model Production Pipeline
==================================================

**pyAMPP** is a Python implementation of the Automatic Model Production Pipeline (AMPP) for solar coronal modeling.  
It streamlines the process of generating realistic 3D solar atmosphere models with minimal user input.


Documentation
-------------

Full documentation is available at:
https://pyampp.readthedocs.io/en/latest/

Overview
--------

**AMPP** automates the production of 3D solar models by:

- Downloading vector magnetic field data from the Helioseismic and Magnetic Imager (HMI) onboard the Solar Dynamics Observatory (SDO)
- Optionally downloading contextual Atmospheric Imaging Assembly (AIA) data
- Performing magnetic field extrapolations (Potential and/or Nonlinear Force-Free Field)
- Generating synthetic plasma emission models assuming either steady-state or impulsive heating
- Producing non-LTE chromospheric models constrained by photospheric measurements
- Enabling interactive 3D inspection and customization through user-friendly GUIs


Installation
------------

pyAMPP can be installed directly from PyPI.

0.  **Setting up a Python 3.10 Environment (Recommended)**

We **strongly recommend** running **pyAMPP** in a dedicated Python 3.10 environment.
You may use any of the following tools to install Python and create an isolated environment:
 - Miniforge: https://github.com/conda-forge/miniforge
 - Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
 - Anaconda: https://www.anaconda.com/docs/getting-started/anaconda/install
 - Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
 - pyenv: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation

Please refer to the official installation instructions for your chosen tool. Their documentation is comprehensive and up to date, so we will not repeat it here.

*You can skip this environment setup section if you already have a suitable Python environment.*

The instructions below use Conda as an example:

1. **Install Conda**
   If you don’t have Conda yet, follow the instructions for your platform here:
   https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2. **Create and activate your environment**
   Open an Anaconda Prompt or Command Prompt and run:

   .. code-block:: bash

      conda create -n suncast python=3.10
      conda activate suncast

3. **Upgrade pip and install pyampp**

   .. code-block:: bash

      pip install -U pip setuptools wheel
      pip install -U pyampp

4. **(Optional) Install additional dependencies**

   For a full scientific Python stack (e.g., SunPy and related tools):

   .. code-block:: bash

      pip install -U sunpy[all]

Main Interfaces
---------------

pyAMPP installs two GUI applications:

1. **gxampp** – Launches a GUI to select observation time and coordinates. It then invokes `gxbox` to build the 3D model.
2. **gxbox** – Launches a GUI that builds and displays the 3D magnetic field and plasma model. Can be run independently if coordinates are known.

Usage Examples
--------------

**1. Launch the time/coord selector (gxampp)**

.. code-block:: bash

    gxampp

.. image:: docs/images/pyampp_gui.png
    :alt: pyampp GUI screenshot
    :align: center
    :width: 600px

**2. Launch the modeling GUI directly (Gxbox Map Viewer)**

.. code-block:: bash

    gxbox \
      --time "2022-03-30T17:22:37" \
      --coords 34.44988566346035 14.26110705696788 \
      --hgs \
      --box-dims 360 180 200 \
      --box-res 0.729 \
      --pad-frac 0.25 \
      --data-dir /path/to/download_dir \
      --gxmodel-dir /path/to/gx_models_dir \
      --external-box /path/to/boxfile.gxbox

.. image:: docs/images/gxbox_gui.png
    :alt: gxbox GUI screenshot
    :align: center
    :width: 600px

The `Gxbox Map Viewer` GUI automatically downloads the required solar data and builds the 3D model based on the user's input. The resulting model can be visualized in a VTK-based viewer (`Gxbox 3D Viewer`) that supports interactive exploration of the magnetic field structure.

Additionally, users can trace and extract magnetic field lines within the 3D model and send them back to the `gxbox` GUI, where they can be overlaid on solar images for contextual visualization.

.. image:: docs/images/MagFieldViewer_gui.png
    :alt: MagFieldViewer GUI screenshot
    :align: center
    :width: 600px

Notes:

- `--coords` takes two floats, separated by space (no brackets or commas).
- One of `--hpc`, `--hgc`, or `--hgs` must be specified to define the coordinate system.
- Remaining parameters are optional and have default values.

Entrypoints
-----------

After installation, the following commands become available:

- ``gxampp``: Launch the time and location GUI.
- ``gxbox``: Launch the modeling GUI directly with CLI options.

License
-------

Copyright (c) 2024, `SUNCAST <https://github.com/suncast-org/>`_ team. Released under the 3-clause BSD license.
