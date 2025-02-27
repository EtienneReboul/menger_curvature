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

def make_universe(topology_name : str , trajectory_name : str ) -> mda.Universe:
    """Make a Universe with a tubulin monomer structure

    Returns
    -------
    MDAnalysis.core.universe.Universe object
    """
    topology_path = os.path.join(files.TEST_DATA_DIR, topology_name)
    trajectory_path = os.path.join(files.TEST_DATA_DIR, trajectory_name)
    return mda.Universe(topology_path, trajectory_path)

def make_filename(md_name : str, spacing : int , metric : str,chainid : str| None = None) -> str:

    filename : str = ""
    if chainid is None:
        filename=f"{md_name}_spacing_{spacing}_{metric}.npy"
    else:
        filename= f"{md_name}_spacing_{spacing}_chain_{chainid}_{metric}.npy"

    return filename

def retrieve_results( name : str,
                     spacing : int,
                     metrics : str,
                     chainid : str = None ) -> np.ndarray:
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
    if not chainid:
        path= os.path.join(
                files.TEST_DATA_DIR,
                make_filename(name,spacing,metrics, chainid=None))
    else:
        path= os.path.join(
                files.TEST_DATA_DIR,
                make_filename(name,spacing,metrics,chainid))
    return np.load(path)
