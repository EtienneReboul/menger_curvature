"""Tests for the Menger curvature analysis module.

This module contains test cases for the MengerCurvature class and related functions
that calculate the Menger curvature of molecular structures. The tests verify the
correctness of curvature calculations, parameter validation, and geometric computations.

The test suite includes:
- TestMengerCurvature: Class testing the main MengerCurvature analysis functionality
- test_check_spacing: Tests for spacing parameter validation
- test_menger_curvature_function: Tests for the core curvature calculation
- test_compute_triangle_edges: Tests for triangle edge length calculations
- test_compute_triangle_area: Tests for triangle area calculations

Test data is loaded from JSON files containing pre-computed results for validation.
The tests use pytest fixtures and parametrization for comprehensive coverage.

"""
# built-in
import json

# third-party
import pytest
from numpy.testing import assert_allclose
import numpy as np

# package
from menger.analysis.mengercurvature import (MengerCurvature,
                                             menger_curvature,
                                             compute_triangle_edges,
                                             compute_triangle_area,
                                             check_spacing,
                                             check_select)
from menger.tests.utils import make_universe, retrieve_results
from MDAnalysis.exceptions import SelectionError

def get_test_parameters():
    with open("menger/data/MengerCurvature_test_parameters.json", "r", encoding="utf-8") as f:
        return json.load(f)

def compare_results(results, params_dict):
    # Define test cases
    test_cases = [
        ('curvature_array', results.curvature_array),
        ('local_curvatures', results.local_curvatures),
        ('local_flexibilities', results.local_flexibilities)
    ]

    # Test each attribute
    for attr_name, test_value in test_cases:
        expected_value = retrieve_results(
            params_dict['md_name'],
            params_dict['spacing'],
            attr_name,
            chainid=params_dict['chainid'])


        assert_allclose(
        test_value,
        expected_value,
        rtol= params_dict['rtol'],
        err_msg=f"{attr_name} are not as expected for md: "+ \
        f"{params_dict['md_name']} spacing {params_dict['spacing']}",
        verbose=True)


class TestMengerCurvature:

    # fixtures are helpful functions that set up a test
    # See more at https://docs.pytest.org/en/stable/how-to/fixtures.html


    @pytest.mark.parametrize(
        "params_dict",
        get_test_parameters())
    def test_menger_curvature(self, params_dict):

        # make universe
        universe = make_universe(
            params_dict["topology_name"], params_dict["trajectory_name"])

        # run menger curvature analysis

        menger_analyser = MengerCurvature(
            universe, params_dict["select"], params_dict['spacing'])
        menger_analyser.run()
        results = menger_analyser.results

        # compare results
        compare_results(results, params_dict)

    @pytest.mark.parametrize(
        "params_dict",
        get_test_parameters())
    def test_menger_curvature_parallel(self, params_dict):
        """Test that parallel execution gives same results as serial"""

        universe = make_universe(
            params_dict["topology_name"], params_dict["trajectory_name"])

        # Run parallel analysis with 2 workers
        mc_parallel = MengerCurvature(
            universe, params_dict["select"],
            params_dict['spacing'],
            n_workers=1)
        mc_parallel.run_parallel()

        # Compare results
        compare_results(mc_parallel.results, params_dict)


@pytest.mark.parametrize("spacing,n_atoms,expected_error,match", [
    (0, 10, ValueError, "Spacing must be at least 1"),
    (6, 10, ValueError, "Spacing is too large for the number of atoms"),  # n_atoms//2 + 1
    (None, 10, ValueError, "Spacing must be provided"),
    (1.5, 10, TypeError, "Spacing must be an integer"),
    (2, 10, None, None),  # Valid case
])
def test_check_spacing(spacing, n_atoms, expected_error, match):
    if expected_error is None:
        assert check_spacing(spacing, n_atoms) is None
    else:
        with pytest.raises(expected_error, match=match):
            check_spacing(spacing, n_atoms)

@pytest.mark.parametrize("select,universe,expected_error,match", [
    (None, make_universe("tubulin_chain_a.pdb", "tubulin_chain_a.dcd"), ValueError, "Selection string must be provided"),
    (123, make_universe("tubulin_chain_a.pdb", "tubulin_chain_a.dcd"), TypeError, "Selection string must be a string"), 
    ("invalid selection", make_universe("tubulin_chain_a.pdb", "tubulin_chain_a.dcd"), SelectionError, "Unknown selection token: 'invalid'"),
    ("name CA and resid 1", make_universe("tubulin_chain_a.pdb", "tubulin_chain_a.dcd"), ValueError, "Selection string must match at least 3 atoms"),
    ("name CA and segid A", make_universe("tubulin_chain_a.pdb", "tubulin_chain_a.dcd"), ValueError, "Selection string did not match any atoms"),
    ("name CA and chainID A", make_universe("tubulin_chain_a.pdb", "tubulin_chain_a.dcd"), None, None)  # Valid case
])
def test_check_select(select, universe, expected_error, match):
    if expected_error is None:
        assert check_select(universe, select) is None
    else:
        with pytest.raises(expected_error, match=match):
            check_select(universe, select)

def test_menger_curvature_function():
    frame = np.array([[13.31, 34.22, 34.36],
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
                      [56.58, 35.18, 34.74]], dtype=np.float32)

    expected = np.array(
        [0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0., 0.02])
    result = np.round(menger_curvature(frame, 2), 2)
    assert_allclose(
        result, expected, err_msg="Menger curvature function produced unexpected results")


def test_compute_triangle_edges():
    # Test with known triangle coordinates
    frame = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Test for right triangle with spacing=1
    edges = compute_triangle_edges(frame, 0, 1)
    expected = np.array([1.0, np.sqrt(2), 1.0])
    assert_allclose(edges, expected,
                    err_msg="Triangle edges computation incorrect for right triangle")

    # Test for equilateral triangle with spacing=1
    eq_frame = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(0.75), 0]
    ])
    edges = compute_triangle_edges(eq_frame, 0, 1)
    assert_allclose(edges, np.array([1.0, 1.0, 1.0]),
                    err_msg="Triangle edges computation incorrect for equilateral triangle")


def test_compute_triangle_area():
    # Test area of right triangle
    edges = np.array([3.0, 4.0, 5.0])
    area = compute_triangle_area(edges)
    assert_allclose(area, 6.0,
                    err_msg="Area computation incorrect for right triangle")

    # Test area of equilateral triangle
    edges = np.array([2.0, 2.0, 2.0])
    area = compute_triangle_area(edges)
    expected_area = 2 * np.sqrt(3) / 2
    assert_allclose(area, expected_area,
                    err_msg="Area computation incorrect for equilateral triangle")

    # Test area of degenerate triangle (no area)
    edges = np.array([1.0, 1.0, 2.0])  # Collinear points
    area = compute_triangle_area(edges)
    assert_allclose(area, 0.0,
                    err_msg="Area computation incorrect for degenerate triangle")
