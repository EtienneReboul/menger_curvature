"""
This module contains functionality for calculating Menger curvature along trajectories.
The Menger curvature provides information about local geometry by calculating the reciprocal
radius of the circumscribed circle passing through three points. 

The module contains:
- menger_curvature(): numba 
- MengerCurvature: Analysis class that processes trajectories to extract curvature information

The analysis can be applied to any set of points, but is particularly useful for analyzing
protein backbone conformations using C-alpha atoms.

Examples
--------
Calculate Menger curvature for the chain A of a tubulin protein trajectory in serial mode:

    >>> import MDAnalysis as mda
    >>> from menger.analysis.mengercurvature import MengerCurvature
    >>> from menger.tests.utils import make_universe
    >>> md_name = "tubulin_chain_a"
    >>> u = make_universe(
    ...     topology_name=f"{md_name}.pdb",
    ...     trajectory_name=f"{md_name}.dcd",
    ... )
    >>> menger_analyser = MengerCurvature(
    ...     u,
    ...     select="name CA and chainID A",
    ...     spacing=2
    ... )
    >>> menger_analyser.run()
    >>> average_curvature = menger_analyser.results.local_curvatures
    >>> flexibility = menger_analyser.results.local_flexibilities
    >>> menger_curvature = menger_analyser.results.curvature_array

Calculate Menger curvature for the chain A of a tubulin protein trajectory in parallel mode:
    >>> import MDAnalysis as mda
    >>> from menger.analysis.mengercurvature import MengerCurvature
    >>> from menger.tests.utils import make_universe
    >>> md_name = "tubulin_chain_a"
    >>> u = make_universe(
    ...     topology_name=f"{md_name}.pdb",
    ...     trajectory_name=f"{md_name}.dcd",
    ... )
    >>> menger_analyser = MengerCurvature(
    ...     u,
    ...     select="name CA and chainID A",
    ...     spacing=2,
    ...     n_workers=4
    ... )
    >>> menger_analyser.run_parallel()
    >>> average_curvature = menger_analyser.results.local_curvatures
    >>> flexibility = menger_analyser.results.local_flexibilities
    >>> menger_curvature = menger_analyser.results.curvature_array

See Also
--------
MDAnalysis.analysis.base.AnalysisBase : Base class for analysis

References
----------
[1]Menger, Karl. Géométrie générale. 1954. <http://eudml.org/doc/192644>.
[2] J. C. Léger. Menger Curvature and Rectifiability. Annals of Mathematics. 
    1999, 149, 831– 869, DOI: 10.2307/121074
[3]Marien J, Prévost C, Sacquin-Mora S. nP-Collabs: Investigating Counterion-Mediated 
    Bridges in the Multiply Phosphorylated Tau-R2 Repeat.J Chem Inf Model. 
    2024 Aug 26;64(16):6570-6582. doi: 10.1021/acs.jcim.4c00742

"""
from typing import Union, TYPE_CHECKING
import multiprocessing as mp
from functools import partial

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
import numpy as np
from numba import njit

if TYPE_CHECKING:
    from MDAnalysis.core.universe import Universe, AtomGroup


@njit()
def compute_triangle_edges(frame: np.ndarray, i: int, spacing: int) -> np.ndarray:
    """
    Calculate the norm of the edges of a triangle given its vertices.

    Args:
        vertices (numpy.ndarray): Array of shape (3, 3) containing the coordinates of the vertices.

    Returns:
        numpy.ndarray: Array of shape (3,) containing the norm of the edges.
    """

    edges_norm = np.zeros(3)
    vertices = np.zeros((3, 3))

    # Get the coordinates of the vertices of the triangle
    vertices[0] = frame[i]
    vertices[1] = frame[i+spacing]
    vertices[2] = frame[i+2*spacing]

    edges_norm[0] = np.linalg.norm(vertices[0] - vertices[1])
    edges_norm[1] = np.linalg.norm(vertices[1] - vertices[2])
    edges_norm[2] = np.linalg.norm(vertices[2] - vertices[0])

    return edges_norm


@njit()
def compute_triangle_area(edges_norm: np.ndarray) -> float:
    """
    Calculate the area of a triangle given the norm of its edges.

    Args:
        edges_norm (numpy.ndarray): Array of shape (3,) containing the norm of the edges.

    Returns:
        float: Area of the triangle.
    """

    semi_perimeter = np.sum(edges_norm) / 2  # Heron's formula
    triangle_area = np.sqrt(
        semi_perimeter * np.prod(semi_perimeter - edges_norm))

    return triangle_area


@njit()
def menger_curvature(frame: np.ndarray, spacing: int) -> np.ndarray:
    """
    Calculate the Menger Curvature for residues in [spacing+1:N-spacing] in a trajectory 
    where N is the total number of residues (counting from 1 to N).

    Args:
        frame (numpy.ndarray): Frame of coordinates of C-alpha atoms.

    Returns:
        numpy.ndarray: Array of Menger curvature values over time per residue. 
    """

    # initialize array cumulative displacement
    frame_curvature = np.zeros(frame.shape[0]-2*spacing)

    # Loop over residues
    for i in range(frame_curvature.shape[0]):

        # Calculate the norm of the edges of the triangle
        edges_norm = compute_triangle_edges(frame, i, spacing)

        # Calculate the area of the triangle
        triangle_area = compute_triangle_area(edges_norm)

        # Calculate the Menger curvature
        frame_curvature[i] = 4 * triangle_area / np.prod(edges_norm)

    return frame_curvature

def menger_curvature_wrapper( frame_index : int,
                             universe : mda.Universe,
                             atom_sel : str,
                             spacing : int ) -> np.ndarray:
    """Wrapper function for parallel computation of Menger curvature
     on molecular dynamics trajectories.
    This function prepares the data from a specific frame of a molecular dynamics trajectory
    for Menger curvature calculation.
    Parameters
    ----------
    frame_index : int
        Index of the frame to analyze in the trajectory
    universe : MDAnalysis.Universe
        MDAnalysis Universe object containing the trajectory and topology
    atom_sel : str
        MDAnalysis selection string to specify which atoms to analyze
    spacing : int
        Number of atoms to skip between triplet points when calculating curvature
    Returns
    -------
    numpy.ndarray
        Array containing the Menger curvature values calculated for the selected atoms
        at the specified frame
    Notes
    -----
    This wrapper is designed to be used with multiprocessing to parallelize calculations
    across multiple trajectory frames.
    """
    # change frame to the desired index
    _= universe.trajectory[frame_index]
    positions = universe.select_atoms(atom_sel).positions
    return menger_curvature(positions, spacing)


def check_spacing(spacing: int, n_atoms: int) -> None:
    """
    Check if the spacing is valid given the number of atoms.

    Args:
        spacing (int): Spacing between atoms.
        n_atoms (int): Number of atoms in the trajectory.
    """
    if spacing is None:
        raise ValueError("Spacing must be provided. " +
                         "We recommend a spacing of 2 for proteic backbones")
    if not isinstance(spacing, int):
        raise TypeError("Spacing must be an integer")
    if spacing < 1:
        raise ValueError("Spacing must be at least 1")
    if 2*spacing >= n_atoms-1:
        raise ValueError("Spacing is too large for the number of atoms")

def check_select(universe, select):
    """
    Check if the selection string is valid for the universe.
    Args:
        universe (MDAnalysis.core.universe.Universe): Universe object.
        select (str): Selection string.
    """
    if select is None:
        raise ValueError("Selection string must be provided")
    if not isinstance(select, str):
        raise TypeError("Selection string must be a string")
    if len(universe.select_atoms(select)) == 0:
        raise ValueError("Selection string did not match any atoms")
    if len(universe.select_atoms(select)) < 3:
        raise ValueError("Selection string must match at least 3 atoms")


class MengerCurvature(AnalysisBase):
    """
    This class calculates the Menger curvature for a trajectory of atoms or molecules.
    The Menger curvature is computed for each set of three points along the trajectory,
    providing information about the local geometry and bending of the molecular structure.

    Parameters
    ----------
    universe_or_atomgroup: Universe or AtomGroup
        Universe or group of atoms to apply this analysis to.
        If a trajectory is associated with the atoms,
        then the computation iterates over the trajectory.
    select: str
        Selection string for atoms to extract from the input Universe or
        AtomGroup

    Attributes
    ----------
    universe: :class:`~MDAnalysis.core.universe.Universe`
        The universe to which this analysis is applied
    atomgroup: :class:`~MDAnalysis.core.groups.AtomGroup`
        The atoms to which this analysis is applied
    results: :class:`~MDAnalysis.analysis.base.Results`
        results of calculation are stored here, after calling
        :meth:`MengerCurvature.run`
    start: Optional[int]
        The first frame of the trajectory used to compute the analysis
    stop: Optional[int]
        The frame to stop at for the analysis
    step: Optional[int]
        Number of frames to skip between each analyzed frame
    n_frames: int
        Number of frames analysed in the trajectory
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`MengerCurvature.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`MengerCurvature.run`

    Results
    -------
    curvature_array: numpy.ndarray
        Array of shape (n_frames, n_atoms - 2*spacing) storing the Menger curvature 
        for each triplet of points
    local_curvatures: numpy.ndarray
        Mean curvature at each position over all frames
    local_flexibilities: numpy.ndarray 
        Standard deviation of curvature at each position

    Notes
    -----
    The Menger curvature is calculated as the reciprocal of the radius of the
    circumscribed circle passing through three points. For each frame, curvatures
    are calculated for all possible triplets of points with the given spacing.
    The local curvature gives information about the average curvature at each position,
    while the local flexibility indicates the range of curvatures available to the triplet.
    Within the context of the proteic backbone, points would usually correspond to C-alpha carbons
    """
    is_compile_numba = False

    def __init__(
        self,
        universe_or_atomgroup: Union["Universe", "AtomGroup"],
        select: str | None = None,
        spacing: int | None = None,
        n_workers: int |None = None,
        **kwargs
    ):
        # the below line must be kept to initialize the AnalysisBase class!
        super().__init__(universe_or_atomgroup.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can should used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results

        # compile the numba function if it hasn't been compiled yet
        if self.is_compile_numba:
            _ = menger_curvature(
                frame=np.array([[13.31, 34.22, 34.36],
                                [16.89, 33.47, 35.28],
                                [20.4, 34.65, 34.76],
                                [23.99, 33.21, 34.96],
                                [27.52, 34.44, 34.73],
                                [31.27, 33.34, 35.16],
                                [34.95, 34.55, 34.84],
                                [38.57, 33.49, 35.07],
                                [42.11, 34.67, 34.64],
                                [45.72, 33.37, 34.84],
                                [49.49, 34.3, 34.62],
                                [53.24, 33.33, 34.85],
                                [56.58, 35.18, 34.74]],
                               dtype=np.float32),
                spacing=2)
            self.is_compile_numba = True

        self.universe = universe_or_atomgroup.universe
        self.atomgroup = universe_or_atomgroup.select_atoms(select)
        self.sel = select
        self.spacing = spacing
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()-2

        # Check if the spacing is valid
        check_spacing(spacing, self.atomgroup.n_atoms)

        # check if the selection is valid
        check_select(self.universe, select)

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        # This is an optional method that runs before
        # _single_frame loops over the trajectory.
        # It is useful for setting up results arrays
        # For example, below we create an array to store
        # the number of atoms with negative coordinates
        # in each frame.
        self.results.curvature_array = np.zeros(
            (self.n_frames, self.atomgroup.n_atoms - 2*self.spacing),
            dtype=np.float32,  # max should 1 Angström  so float32 is
            # enough even with mean and standard deviation porbably float16
            # would work too
        )
        self.results.local_curvatures = np.zeros(
            self.atomgroup.n_atoms - 2*self.spacing)
        self.results.local_flexibilities = np.zeros(
            self.atomgroup.n_atoms - 2*self.spacing)

    def _single_frame(self):
        """Calculate data from a single frame of trajectory"""
        # This runs once for each frame of the trajectory
        # It can contain the main analysis method, or just collect data
        # so that analysis can be done over the aggregate data
        # in _conclude.

        # The trajectory positions update automatically
        coords = self.atomgroup.positions


        self.results.curvature_array[self._frame_index] = menger_curvature(
            coords, self.spacing)

    def run_parallel(self):
        """Run the analysis in parallel using multiprocessing.pool"""

        # self._prepare()

        # make partial object for menger analysis
        mcc = partial(menger_curvature_wrapper,
                      universe= self.universe,
                      atom_sel=self.sel,
                      spacing=self.spacing
                      )

        #  do pool parralelization
        with mp.Pool(self.n_workers) as pool:
            results = pool.map(mcc, range(self._trajectory.n_frames))

            self.results.curvature_array = np.array(results, dtype=np.float32)
            self._conclude()

    def _conclude(self):
        """Calculate the final results of the analysis"""

        self.results.local_curvatures = np.mean(
            self.results.curvature_array, axis=0)
        self.results.local_flexibilities = np.std(
            self.results.curvature_array, axis=0)
