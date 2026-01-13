from dataclasses import dataclass
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from MDAnalysis.analysis.base import Results


@dataclass
class HeatmapConfig:
    """Configuration for heatmap plots."""

    cmap: str = "Blues"
    vmin: float = 0.0
    vmax: float = 0.40


@dataclass
class SecondaryStructureConfig:
    """Configuration for secondary structure visualization."""

    range: tuple[float, float]
    color: str
    alpha: float = 0.2


@dataclass
class FigureConfig:
    """Configuration for basic figure settings."""

    figsize: tuple[int, int] = (12, 8)
    font_size: int = 12
    style: str = "default"


@dataclass
class PlotConfig:
    """Configuration for plot formatting parameters."""

    figure: FigureConfig = None
    heatmap: HeatmapConfig = None
    beta_sheet: SecondaryStructureConfig = None
    alpha_helix: SecondaryStructureConfig = None
    flexibility_threshold_color: str = "purple"
    flexibility_threshold_linestyle: str = "--"

    def __post_init__(self):
        if self.figure is None:
            self.figure = FigureConfig()
        if self.heatmap is None:
            self.heatmap = HeatmapConfig()
        if self.beta_sheet is None:
            self.beta_sheet = SecondaryStructureConfig(range=(0.01, 0.07), color="blue")
        if self.alpha_helix is None:
            self.alpha_helix = SecondaryStructureConfig(range=(0.28, 0.32), color="red")


class MengerCurvaturePlotter:
    """A class to handle all standard plotting for Menger curvature analysis."""

    def __init__(
        self,
        menger_results: Results,
        spacing: int,
        config: PlotConfig | None = None,
    ):
        """
        Initialize the plotter.

        Args:
            menger_results: Results object from Menger curvature analysis
            spacing: Spacing parameter used in the analysis
            config: PlotConfig instance for plot formatting parameters
        """
        # Store reference to results
        self.menger_results = menger_results
        self.spacing = spacing

        # Use provided config or create default
        self.config = config if config is not None else PlotConfig()

        # compute some useful parameters
        self.number_of_conformations = self.menger_results.curvature_array.shape[0]
        self.range_frame_number = range(self.number_of_conformations)
        self.number_of_atoms_in_selection = self.menger_results.curvature_array.shape[1]
        self.range_residues = range(
            self.spacing + 1, self.number_of_atoms_in_selection + self.spacing + 1
        )

        # Set plotting style
        plt.style.use(self.config.figure.style)

    def plot_curvature_heatmap(self, figsize: None | tuple[int, int] = None):
        """
        Plot Menger curvature as a heatmap across frames and residues.

        Args:
            figsize: Figure size for the plot for overriding default
        """

        # Set figure size to default if not provided
        if figsize is None:
            figsize = self.config.figure.figsize

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()

        ax.set_xlabel("Frames", fontsize=self.config.figure.font_size)
        ax.set_ylabel("Residues", fontsize=self.config.figure.font_size)

        ax.yaxis.set_major_locator(mticker.FixedLocator(self.range_residues))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(self.range_residues))

        ax.yaxis.get_label().set_fontsize(self.config.figure.font_size)
        ax.set_ylim([0.5, self.number_of_atoms_in_selection + 2 * self.spacing + 0.5])

        im = ax.pcolormesh(
            self.range_frame_number,
            self.range_residues,
            np.transpose(self.menger_results.curvature_array),
            cmap=self.config.heatmap.cmap,
            shading="nearest",
            vmin=self.config.heatmap.vmin,
            vmax=self.config.heatmap.vmax,
        )

        cbar = plt.colorbar(im)
        cbar.set_label(label="Menger Curvature ($Å^{-1}$)", fontsize=15, weight="bold")

        return fig

    def plot_local_curvature(
        self, figsize: None | tuple[int, int] = None, legend_loc: str = "upper right"
    ):
        """
        Plot local curvature across residues.
        Args:
            figsize: Figure size for the plot for overriding default
        """
        if figsize is None:
            figsize = self.config.figure.figsize
        fig = plt.figure(figsize=figsize)
        plt.xlim(0.5, self.number_of_atoms_in_selection + 2 * self.spacing + 0.5)

        plt.plot(self.range_residues, self.menger_results.local_curvatures)

        plt.xlabel("Residues", fontsize=self.config.figure.font_size)
        plt.ylabel("Local Curvature ($Å^{-1}$)", fontsize=self.config.figure.font_size)

        # Indicate the range for beta-sheets
        plt.fill_between(
            range(-1, self.number_of_atoms_in_selection + 2 * self.spacing + 10),
            y1=self.config.beta_sheet.range[0],
            y2=self.config.beta_sheet.range[1],
            color=self.config.beta_sheet.color,
            alpha=self.config.beta_sheet.alpha,
            label="β-sheet",
        )

        # Indicate the range for alpha-helices
        plt.fill_between(
            range(-1, self.number_of_atoms_in_selection + 2 * self.spacing + 10),
            y1=self.config.alpha_helix.range[0],
            y2=self.config.alpha_helix.range[1],
            color=self.config.alpha_helix.color,
            alpha=self.config.alpha_helix.alpha,
            label="α-helix",
        )

        plt.legend(loc=legend_loc)

        return fig

    def plot_local_flexibility(
        self, figsize: None | tuple[int, int] = None, threshold: float | None = 0.07
    ):
        """Plot local flexibility across residues.
        Args:
            figsize: Figure size for the plot for overriding default
            threshold: Flexibility threshold to indicate on the plot
            if None, no threshold line is drawn
        """

        if figsize is None:
            figsize = self.config.figure.figsize
        fig = plt.figure(figsize=figsize)
        plt.xlim(0.5, self.number_of_atoms_in_selection + 2 * self.spacing + 0.5)

        plt.plot(self.range_residues, self.menger_results.local_flexibilities)

        plt.xlabel("Residues", fontsize=self.config.figure.font_size)
        plt.ylabel("Local Flexibility ($Å^{-1}$)", fontsize=self.config.figure.font_size)

        if threshold is not None:
            plt.axhline(
                y=threshold,
                linestyle=self.config.flexibility_threshold_linestyle,
                color=self.config.flexibility_threshold_color,
            )

        return fig
