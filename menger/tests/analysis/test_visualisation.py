import numpy as np
import pytest
from menger.analysis.visualisation import MengerCurvaturePlotter


# Mock Results class
class MockResults:
    def __init__(self):
        self.curvature_array = np.random.rand(500, 448)  # 500 frames, 448 atoms
        self.local_curvatures = np.random.rand(448)  # 448 atoms
        self.local_flexibilities = np.random.rand(448)  # 448 atoms

@pytest.fixture
def mock_results():
    return MockResults()

def test_plot_curvature_heatmap(mock_results):
    plotter = MengerCurvaturePlotter(mock_results, spacing=2)
    fig = plotter.plot_curvature_heatmap()
    
    assert fig is not None
    assert len(fig.axes) == 2  # Check if two axes are created (heatmap + colorbar)
    
    # Check properties of the heatmap axis
    heatmap_axis = fig.axes[0]
    assert heatmap_axis.get_xlabel() == 'Frames'
    assert heatmap_axis.get_ylabel() == 'Residues'
    
    # Check properties of the colorbar axis
    colorbar_axis = fig.axes[1]
    assert colorbar_axis.get_label() == '<colorbar>'

def test_plot_local_curvature(mock_results):
    plotter = MengerCurvaturePlotter(mock_results, spacing=2)
    fig = plotter.plot_local_curvature()
    
    assert fig is not None
    assert len(fig.axes) == 1  # Check if one axis is created


def test_plot_local_flexibility(mock_results):
    plotter = MengerCurvaturePlotter(mock_results, spacing=2)
    fig = plotter.plot_local_flexibility()
    
    assert fig is not None
    assert len(fig.axes) == 1  # Check if one axis is created
    

def test_plot_local_flexibility_with_threshold(mock_results):
    plotter = MengerCurvaturePlotter(mock_results, spacing=2)
    fig = plotter.plot_local_flexibility(threshold=0.1)
    
    assert fig is not None
    assert len(fig.axes) == 1  # Check if one axis is created
    assert any(line.get_ydata()[0] == 0.1 for line in fig.axes[0].lines)  # Check if threshold line is drawn