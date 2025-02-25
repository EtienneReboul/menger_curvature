"""This module provides utility functions for creating test universes in MDAnalysis.


The module contains functions to create Universe objects for testing purposes,
particularly focused on tubulin structures and dummy reference universes.

Functions
---------
make_tubulin_monomer_universe()
  Creates an MDAnalysis Universe containing a tubulin monomer structure
retrieve_results(name, spacing, metrics)
  Retrieves stored Menger curvature analysis results from a file

Notes
-----
This module is intended to be used in testing and development of the Menger curvature
analysis module in MDAnalysis.

"""

# Standard library imports
import os

# Third party imports
import MDAnalysis as mda
import numpy as np

# package imports
from menger.data import files

def make_tubulin_monomer_universe():
    """Make a Universe with a tubulin monomer structure

    Returns
    -------
    MDAnalysis.core.universe.Universe object
    """
    return mda.Universe(files.TUBULIN_CHAIN_A_PDB, files.TUBULIN_CHAIN_A_DCD)

def retrieve_results( name : str, spacing : int, metrics : str ,segid : str = None ) -> np.ndarray:
    """Retrieve the results of the Menger curvature analysis from a file

    Parameters
    ----------
    name : str
        The name of the file to retrieve the results from
    spacing : int
        The spacing used in the analysis
    n_frames : int
        The number of frames in the trajectory

    Returns
    -------
    np.ndarray
        The results of the analysis
    """
    if not segid:
        path= os.path.join(
                files.TEST_DATA_DIR,
                f"{name}_spacing_{spacing}_{metrics}.npy")
    else:
        path= os.path.join(
                files.TEST_DATA_DIR,
                f"{name}_spacing_{spacing}_segid_{segid}_{metrics}.npy")
    return np.load(path)
        
