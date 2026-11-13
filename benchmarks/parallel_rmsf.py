"""Split-apply-combine RMSF pipeline usable with MDAnalysis parallel backends.

MDAnalysis 2.10 only runs ``AverageStructure``, ``AlignTraj`` and ``RMSF``
serially. To benchmark RMSF with the same backends and multiprocessing start
methods as the Menger curvature, this module implements the same pipeline as
two parallelisable ``AnalysisBase`` passes:

1. :class:`AverageStructureCA` fits every frame onto a reference frame
   (by default frame 0) and averages the fitted coordinates of the selection.
   This is what ``align.AverageStructure(u, u, select=sel, ref_frame=0)`` does.
2. :class:`AlignedRMSF` fits every frame onto that average structure and
   accumulates the per-atom mean and variance (Welford). Each worker returns
   ``(n, mean, M2)`` for its block of frames, and the blocks are merged with
   Chan et al.'s parallel formula. This replaces
   ``align.AlignTraj(..., in_memory=True)`` followed by ``rms.RMSF``.

The fit (centre of geometry plus QCP rotation, no weights) is the one that
MDAnalysis uses. Only the selected atoms are transformed and nothing is kept
in memory beyond O(N_sel) accumulators, unlike ``AlignTraj(in_memory=True)``,
which stores the whole aligned trajectory of all atoms. Results agree with the
MDAnalysis pipeline to float32 precision (see ``check_against_mdanalysis``).
"""
from __future__ import annotations

import numpy as np
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup


def _fit(positions: np.ndarray, ref_centered: np.ndarray, ref_com: np.ndarray) -> np.ndarray:
    """Superimpose ``positions`` on the reference (unweighted, like MDAnalysis' default)."""
    positions = positions.astype(np.float64)
    centered = positions - positions.mean(axis=0)
    rot, _ = rotation_matrix(centered, ref_centered)
    return centered @ rot.T + ref_com


def _sum(values):
    return sum(values)


def _merge_welford(blocks):
    """Merge per-block (n, mean, M2) statistics (Chan, Golub & LeVeque 1979)."""
    n_tot, mean_tot, m2_tot = blocks[0]
    n_tot = float(n_tot[0])
    mean_tot = mean_tot.copy()
    m2_tot = m2_tot.copy()
    for n_arr, mean, m2 in blocks[1:]:
        n = float(n_arr[0])
        if n == 0:
            continue
        delta = mean - mean_tot
        n_new = n_tot + n
        mean_tot += delta * (n / n_new)
        m2_tot += m2 + delta**2 * (n_tot * n / n_new)
        n_tot = n_new
    return [np.array([n_tot]), mean_tot, m2_tot]


class _ParallelisableBase(AnalysisBase):
    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")


class AverageStructureCA(_ParallelisableBase):
    """Average coordinates of ``atomgroup`` after fitting each frame onto ``ref_frame``."""

    def __init__(self, atomgroup, ref_frame: int = 0, **kwargs):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup
        traj = atomgroup.universe.trajectory
        current = traj.ts.frame
        try:
            traj[ref_frame]
            ref = atomgroup.positions.astype(np.float64)
        finally:
            traj[current]
        self._ref_com = ref.mean(axis=0)
        self._ref_centered = ref - self._ref_com

    def _prepare(self):
        self.results.coord_sum = np.zeros((self._ag.n_atoms, 3))
        self.results.count = 0

    def _single_frame(self):
        self.results.coord_sum += _fit(self._ag.positions, self._ref_centered, self._ref_com)
        self.results.count += 1

    def _get_aggregator(self):
        return ResultsGroup(lookup={"coord_sum": ResultsGroup.ndarray_sum, "count": _sum})

    def _conclude(self):
        self.results.positions = self.results.coord_sum / self.results.count


class AlignedRMSF(_ParallelisableBase):
    """RMSF of ``atomgroup`` after fitting every frame onto ``ref_positions``."""

    def __init__(self, atomgroup, ref_positions: np.ndarray, **kwargs):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup
        ref = np.asarray(ref_positions, dtype=np.float64)
        self._ref_com = ref.mean(axis=0)
        self._ref_centered = ref - self._ref_com

    def _prepare(self):
        # list of arrays updated in place, so that the per-worker block
        # statistics are always available to the aggregator
        self.results.stats = [
            np.zeros(1),
            np.zeros((self._ag.n_atoms, 3)),
            np.zeros((self._ag.n_atoms, 3)),
        ]

    def _single_frame(self):
        n, mean, m2 = self.results.stats
        x = _fit(self._ag.positions, self._ref_centered, self._ref_com)
        n += 1
        delta = x - mean
        mean += delta / n[0]
        m2 += delta * (x - mean)

    def _get_aggregator(self):
        return ResultsGroup(lookup={"stats": _merge_welford})

    def _conclude(self):
        n, _, m2 = self.results.stats
        self.results.rmsf = np.sqrt(m2.sum(axis=1) / n[0])


def check_against_mdanalysis(topology, trajectory, select="protein and name CA", **run_kwargs):
    """Return the max absolute RMSF difference (Å) between this pipeline and MDAnalysis'."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms

    u = mda.Universe(topology, trajectory)
    average = align.AverageStructure(u, u, select=select, ref_frame=0).run()
    align.AlignTraj(u, average.results.universe, select=select, in_memory=True).run()
    reference = rms.RMSF(u.select_atoms(select)).run().results.rmsf

    u2 = mda.Universe(topology, trajectory)
    ca = u2.select_atoms(select)
    avg = AverageStructureCA(ca).run(**run_kwargs)
    ours = AlignedRMSF(ca, avg.results.positions).run(**run_kwargs).results.rmsf
    return float(np.max(np.abs(ours - reference)))
