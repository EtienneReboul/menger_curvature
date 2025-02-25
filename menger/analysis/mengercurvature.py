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
Calculate Menger curvature for a protein trajectory:

    >>> import MDAnalysis as mda
    >>> from menger.analysis import MengerCurvature
    >>> u = mda.Universe("protein.pdb", "trajectory.xtc") 
    >>> calpha = u.select_atoms("name CA")
    >>> MC = MengerCurvature(calpha)
    >>> MC.run()
    >>> average_curvature = MC.results.local_curvatures
    >>> flexibility = MC.results.local_flexibilities

See Also
--------
MDAnalysis.analysis.base.AnalysisBase : Base class for analysis

References
.. [1] Lewiner, T., Gomes Jr, J., Lopes, H., & Craizer, M. (2005). 
       Curvature and torsion estimators based on parametric curve fitting. 
       Computers & Graphics, 29(5), 641-655.

"""
from typing import Union, TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
import numpy as np
from numba import njit

if TYPE_CHECKING:
    from MDAnalysis.core.universe import Universe, AtomGroup


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

        # loop variables
        vertices = np.zeros((3, 3))
        edges_norm = np.zeros(3)

        # Get the coordinates of the vertices of the triangle
        vertices[0] = frame[i]
        vertices[1] = frame[i+spacing]
        vertices[2] = frame[i+2*spacing]

        # Calculate the norm of the edges
        edges_norm[0] = np.linalg.norm(vertices[0] - vertices[1])
        edges_norm[1] = np.linalg.norm(vertices[1] - vertices[2])
        edges_norm[2] = np.linalg.norm(vertices[2] - vertices[0])

        # Calculate the area of the triangle
        semi_perimeter = np.sum(edges_norm) / 2  # Heron's formula
        triangle_area = np.sqrt(
            semi_perimeter * np.prod(semi_perimeter - edges_norm))

        # Calculate the Menger curvature
        frame_curvature[i] = 4 * triangle_area / np.prod(edges_norm)

    return frame_curvature


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
    menger_array: numpy.ndarray
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
    menger_curvature = None

    def __init__(
        self,
        universe_or_atomgroup: Union["Universe", "AtomGroup"],
        select: str = "all",
        spacing: int = 2,
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
        if MengerCurvature.menger_curvature is None:
            MengerCurvature.menger_curvature = menger_curvature
            MengerCurvature.menger_curvature(
                frame= np.array([[13.31, 34.22, 34.36],
                                [16.89, 33.47, 35.28],
                                [20.4 , 34.65, 34.76],
                                [23.99, 33.21, 34.96],
                                [27.52, 34.44, 34.73],
                                [31.27, 33.34, 35.16],
                                [34.95, 34.55, 34.84],
                                [38.57, 33.49, 35.07],
                                [42.11, 34.67, 34.64],
                                [45.72, 33.37, 34.84],
                                [49.49, 34.3 , 34.62],
                                [53.24, 33.33, 34.85],
                                [56.58, 35.18, 34.74]],
                                dtype=np.float32),
                spacing=2)


        self.universe = universe_or_atomgroup.universe
        self.atomgroup = universe_or_atomgroup.select_atoms(select)
        self.spacing = spacing

        if spacing < 1:
            raise ValueError("Spacing must be at least 1")
        elif 2*spacing >= self.atomgroup.n_atoms-1:
            raise ValueError("Spacing is too large for the number of atoms")

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        # This is an optional method that runs before
        # _single_frame loops over the trajectory.
        # It is useful for setting up results arrays
        # For example, below we create an array to store
        # the number of atoms with negative coordinates
        # in each frame.
        self.results.menger_array = np.zeros(
            (self.n_frames, self.atomgroup.n_atoms - 2*self.spacing),
            dtype=np.float32,  # max should 1 Angström  so float32 is
            # enough even with mean and standard deviation porbably float16
            # would work too
        )
        self.results.local_curvatures = np.zeros(self.atomgroup.n_atoms - 2*self.spacing)
        self.results.local_flexibilities = np.zeros(self.atomgroup.n_atoms - 2*self.spacing)

    def _single_frame(self):
        """Calculate data from a single frame of trajectory"""
        # This runs once for each frame of the trajectory
        # It can contain the main analysis method, or just collect data
        # so that analysis can be done over the aggregate data
        # in _conclude.

        # The trajectory positions update automatically
        coords = self.atomgroup.positions
        # You can access the frame number using self._frame_index
        # self.results.menger_array[self._frame_index] = menger_curvature(
        #    coords, self.spacing)

        self.results.menger_array[self._frame_index] = MengerCurvature.menger_curvature(
            coords, self.spacing)


    def _conclude(self):
        """Calculate the final results of the analysis"""

        self.results.local_curvatures = np.mean(
            self.results.menger_array, axis=0)
        self.results.local_flexibilities = np.std(
            self.results.menger_array, axis=0)
