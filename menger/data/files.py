"""
This module provides paths to example data files used in the menger package.

The data files include:
- MDAnalysis logo text file
- Tubulin chain A trajectory in DCD format  
- Tubulin chain A structure in PDB format

Usage
-----
Import all file paths::


Available Files
--------------
MDANALYSIS_LOGO : Path
    Path to MDAnalysis logo text fileTUBULIN_CHAIN_A_DCD : Path  
    Path to example DCD trajectory file of tubulin chain A
TUBULIN_CHAIN_A_PDB : Path
    Path to example PDB structure file of tubulin chain A

Notes
-----
The files are accessed using Python's importlib.resources to ensure 
platform-independent and installation-safe file access.
"""

__all__ = [
    "MDANALYSIS_LOGO",  # example file of MDAnalysis logo
    "TUBULIN_CHAIN_A_DCD",  # example DCD file
    "TUBULIN_CHAIN_A_PDB",  # example PDB file
    "TUBULIN_CHAIN_ABC_DCD",  # example DCD file
    "TUBULIN_CHAIN_ABC_PDB",  # example PDB file
    "TEST_DATA_DIR",  # directory containing test data
]

import importlib.resources

data_directory = importlib.resources.files("menger") / "data"

MDANALYSIS_LOGO = data_directory / "mda.txt"
TUBULIN_CHAIN_A_DCD = data_directory / "tubulin_chain_A.dcd"
TUBULIN_CHAIN_A_PDB = data_directory / "tubulin_chain_A.pdb"
TUBULIN_CHAIN_ABC_DCD= data_directory / "tubulin_chain_ABC.dcd"
TUBULIN_CHAIN_ABC_PDB= data_directory / "tubulin_chain_ABC.pdb"
TEST_DATA_DIR = data_directory / "test_data"
