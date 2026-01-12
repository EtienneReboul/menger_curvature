import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from MDAnalysis.analysis.base import Results


class MengerCurvaturePlotter:
    """A class to handle all standard plotting for Menger curvature analysis."""
    
    def __init__(self, menger_results : Results, spacing :int, figsize : tuple[int, int]=(12, 8), font_size : int=12, style: str='default'):
        """
        Initialize the plotter.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        # Store reference to results
        self.menger_curvature = menger_results.curvature_array
        self.local_curvatures = menger_results.local_curvatures
        self.local_flexibilities = menger_results.local_flexibilities
        self.spacing = spacing

        # compute some useful parameters
        self.number_of_conformations = self.menger_curvature.shape[0]
        self.range_frame_number = range(self.number_of_conformations)
        self.number_of_atoms_in_selection = self.menger_curvature.shape[1]
        self.range_residues = range(self.spacing + 1, self.number_of_atoms_in_selection + self.spacing + 1)
        
        # Set plotting parameters
        self.font_size = font_size
        self.figsize = figsize
        self.style = style
        plt.style.use(style)

    def plot_curvature_heatmap(self,  figsize : None | tuple[int, int]=None):
        """
        Plot Menger curvature as a heatmap across frames and residues.
        
        Args:
            figsize: Figure size for the plot for overriding default
        """
        
        
        # Set figure size to default if not provided
        if figsize is None:
            figsize = self.figsize
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        
        ax.set_xlabel("Frames", fontsize=self.font_size)
        ax.set_ylabel("Residues", fontsize=self.font_size)
        
        ax.yaxis.set_major_locator(mticker.FixedLocator(self.range_residues))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(self.range_residues))
        
        ax.yaxis.get_label().set_fontsize(self.font_size)
        ax.set_ylim([0.5, self.number_of_atoms_in_selection + 2 * self.spacing + 0.5])
        
        im = ax.pcolormesh(self.range_frame_number, self.range_residues, np.transpose(self.menger_curvature), 
                   cmap="Blues", shading='nearest', vmin=0.0, vmax=0.40)
        
        cbar = plt.colorbar(im)
        cbar.set_label(label="Menger Curvature ($Å^{-1}$)", fontsize=15, weight='bold')
        
        return fig
    
    def plot_local_curvature(self, figsize : None | tuple[int, int]=None, legend_loc: str='upper right'):
        """
        Plot local curvature across residues.
        Args:
            figsize: Figure size for the plot for overriding default
        """
        if figsize is None:
            figsize = self.figsize
        fig=plt.figure(figsize=figsize
                       )
        plt.xlim(0.5, self.number_of_atoms_in_selection+2*self.spacing+0.5)

        plt.plot(self.range_residues, self.local_curvatures)

        plt.xlabel("Residues", fontsize=self.font_size)
        plt.ylabel("Local Curvature ($Å^{-1}$)", fontsize=self.font_size)

        # Indicate the range for beta-sheets
        plt.fill_between(range(-1,self.number_of_atoms_in_selection+2*self.spacing+10), y1=0.01, y2=0.07, color="blue", alpha=0.2, label="β-sheet")

        # Indicate the range for alpha-helices
        plt.fill_between(range(-1,self.number_of_atoms_in_selection+2*self.spacing+10), y1=0.28, y2=0.32, color="red", alpha=0.2, label="α-helix")

        plt.legend(loc=legend_loc)

        return fig
    
    def plot_local_flexibility(self, figsize : None | tuple[int, int]=None, threshold: float | None=0.07):
        """Plot local flexibility across residues.
        Args:
            figsize: Figure size for the plot for overriding default
            threshold: Flexibility threshold to indicate on the plot
            if None, no threshold line is drawn
        """
        
        if figsize is None:
            figsize = self.figsize
        fig=plt.figure(figsize=figsize
                       )
        plt.xlim(0.5, self.number_of_atoms_in_selection+2*self.spacing+0.5)

        plt.plot(self.range_residues, self.local_flexibilities)

        plt.xlabel("Residues", fontsize=self.font_size)
        plt.ylabel("Local Flexibility ($Å^{-1}$)", fontsize=self.font_size)

        if threshold is not None:
            plt.axhline(y=threshold, linestyle="--", color="purple")

        return fig
    
