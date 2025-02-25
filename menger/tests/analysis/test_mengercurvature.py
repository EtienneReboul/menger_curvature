import pytest
from numpy.testing import assert_allclose
import numpy as np

from menger.analysis.mengercurvature import MengerCurvature, menger_curvature
from menger.tests.utils import make_tubulin_monomer_universe, retrieve_results


class TestMengerCurvature:

    # fixtures are helpful functions that set up a test
    # See more at https://docs.pytest.org/en/stable/how-to/fixtures.html
    @pytest.fixture
    def universe(self):
        u = make_tubulin_monomer_universe()
        return u
    @pytest.fixture
    def select(self):
        return "name CA"
    @pytest.fixture
    def md_name(self):
        return "tubulin_chain_a"
    
    @pytest.mark.parametrize(
    "spacing",
    [
        1,
        2,
        3,
        4,
        5
    ]
    )
    def test_menger_curvature(self, md_name  : str,universe: np.ndarray,select : str, spacing: int):
        menger_analyser = MengerCurvature(universe, select, spacing)
        menger_analyser.run()
        results = menger_analyser.results
        
        # test menger array 
        test_menger_array = results.menger_array
        expected_menger_array = retrieve_results(md_name, spacing, "menger_array")
        assert_allclose(test_menger_array,
                        expected_menger_array,
                        err_msg="Menger array is not as expected" + \
                        f" for md: {md_name} spacing {spacing}"
                        )

        # test local curvatures
        test_local_curvatures = results.local_curvatures
        expected_local_curvatures = retrieve_results(md_name, spacing, "local_curvatures")
        assert_allclose(test_local_curvatures,
                        expected_local_curvatures,
                        err_msg="Local curvatures are not as expected"+ \
                        f" for md: {md_name} spacing {spacing}"
                        )

        # test local flexibilities
        test_local_flexibilities = results.local_flexibilities
        expected_local_flexibilities = retrieve_results(md_name, spacing, "local_flexibilities")
        assert_allclose(test_local_flexibilities,
                        expected_local_flexibilities,
                        err_msg="Local flexibilities are not as expected"+ \
                        f" for md: {md_name} spacing {spacing}"
                        )
    
    def test_invalid_spacing_values(self, universe, select):
        # Test spacing < 1
        with pytest.raises(ValueError, match="Spacing must be at least 1"):
            menger_analyser = MengerCurvature(universe, select, spacing=0)
        
        # Test spacing too large
        n_atoms = universe.select_atoms(select).n_atoms
        too_large_spacing = n_atoms // 2 + 1
        with pytest.raises(ValueError, match="Spacing is too large for the number of atoms"):
            menger_analyser = MengerCurvature(universe, select, spacing=too_large_spacing)

    
    def test_menger_curvature_function(self):
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

        expected = np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.  , 0.02])
        result = np.round(menger_curvature(frame, 2), 2)
        assert_allclose(result, expected, err_msg="Menger curvature function produced unexpected results")

