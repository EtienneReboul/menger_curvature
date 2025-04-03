.. menger_curvature documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Menger_Curvature Documentation
==============================

Menger_Curvature is a Python package that calculates the Menger curvature of polymeric backbones along molecular trajectories, particularly useful for analyzing protein backbone conformations.

Scientific Background
---------------------
The Menger curvature is a geometric measure that characterizes how much a curve passing through 3 points deviates from being straight. For a triplet of points, it is calculated as the reciprocal of the radius of the circumscribed circle passing through these points.

In protein analysis, this metric provides:
Local geometric information about backbone conformation
Insights into structural flexibility when analyzed across trajectories
Quantitative measures for comparing different protein regions of the backbone

Computational Approach
----------------------
The package implements:
JIT-accelerated calculations using Numba for performance
Parallel computation support for large trajectories
Integration with MDAnalysis for efficient trajectory handling
Memory-efficient array operations with NumPy

Installation
------------

Quick installation using pip:

.. code-block:: bash

   pip install menger-curvature

For development installation:

.. code-block:: bash

   git clone https://github.com/EtienneReboul/menger_curvature.git
   cd menger_curvature
   pip install -e .

Basic Usage
-----------

Calculate Menger curvature for a protein trajectory:

.. code-block:: python

   import MDAnalysis as mda
   from menger.analysis.mengercurvature import MengerCurvature
   from menger.data import files

   # replace by your own filepaths 
   topology = files.TUBULIN_CHAIN_A_PDB 
   trajectory = files.TUBULIN_CHAIN_A_DCD
   u = mda.Universe(topology, trajectory)

   # run analysis in serial mode 
   menger_analyser = MengerCurvature(
      u,
      select="name CA and chainID A",
      spacing=2
      )
   menger_analyser.run()

   # retrieve results data
   average_curvature = menger_analyser.results.local_curvatures
   flexibility = menger_analyser.results.local_flexibilities
   menger_curvature = menger_analyser.results.curvature_array

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
