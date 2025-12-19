import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection, AcqusitionAxesName
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters

from deep_ccf_registration.utils.transforms import (
    get_cropped_region_from_array,
    transform_points_to_template_ants_space,
    apply_transforms_to_points,
    map_points_to_right_hemisphere,
)
from deep_ccf_registration.metadata import AcquisitionAxis


class TestGetCroppedRegionFromArray:
    """Tests for the get_cropped_region_from_array function."""

    def test_basic_cropping(self):
        """Test basic array cropping with points."""
        # Create a 10x10x10 array
        array = np.arange(1000).reshape(10, 10, 10, 1)

        # Points spanning from (2,2,2) to (5,5,5)
        points = np.array([
            [2.0, 2.0, 2.0],
            [5.0, 5.0, 5.0],
        ])

        cropped = get_cropped_region_from_array(
            array=array,
            points=points,
            padding=0
        )

        # Should crop to [2:5, 2:5, 2:5] (floor min to ceil max, exclusive end)
        expected_shape = (3, 3, 3, 1)
        assert cropped.shape == expected_shape

    def test_with_padding(self):
        """Test cropping with padding."""
        array = np.arange(1000).reshape(10, 10, 10, 1)

        points = np.array([
            [5.0, 5.0, 5.0],
        ])

        cropped = get_cropped_region_from_array(
            array=array,
            points=points,
            padding=2
        )

        # With padding=2, should crop to [3:7, 3:7, 3:7]
        # (floor(5)-2=3 to ceil(5)+2=7, exclusive end)
        expected_shape = (4, 4, 4, 1)
        assert cropped.shape == expected_shape

    def test_clamping_to_array_bounds(self):
        """Test that cropping is clamped to array boundaries."""
        array = np.arange(1000).reshape(10, 10, 10, 1)

        # Points near the edge
        points = np.array([
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ])

        cropped = get_cropped_region_from_array(
            array=array,
            points=points,
            padding=10  # Large padding that would go out of bounds
        )

        # Should be clamped to array boundaries
        # Min would be floor(0.5) - 10 = -10, clamped to 0
        # Max would be ceil(1.0) + 10 = 11, clamped to 10
        assert cropped.shape[0] <= array.shape[0]
        assert cropped.shape[1] <= array.shape[1]
        assert cropped.shape[2] <= array.shape[2]

    def test_points_modified_inplace(self):
        """Test that points are modified in-place to be relative to crop."""
        array = np.arange(1000).reshape(10, 10, 10, 1)

        points = np.array([
            [5.0, 5.0, 5.0],
            [7.0, 7.0, 7.0],
        ], dtype=float)

        original_points = points.copy()

        get_cropped_region_from_array(
            array=array,
            points=points,
            padding=1
        )

        # Points should now be relative to the cropped region
        # Original min was floor(5.0)-1 = 4
        # So points should be shifted by -4 in each dimension
        expected_shift = np.array([4.0, 4.0, 4.0])
        np.testing.assert_array_equal(
            points,
            original_points - expected_shift
        )

    def test_points_outside_bounds_raises_error(self):
        """Test that completely out-of-bounds points raise ValueError."""
        array = np.arange(1000).reshape(10, 10, 10, 1)

        # Points completely outside the array
        points = np.array([
            [20.0, 20.0, 20.0],
            [25.0, 25.0, 25.0],
        ])

        with pytest.raises(ValueError, match="Points are completely outside array"):
            get_cropped_region_from_array(
                array=array,
                points=points,
                padding=0
            )

    def test_fractional_coordinates(self):
        """Test handling of fractional coordinates."""
        array = np.arange(1000).reshape(10, 10, 10, 1)

        points = np.array([
            [2.3, 2.7, 2.1],
            [5.8, 5.2, 5.9],
        ])

        cropped = get_cropped_region_from_array(
            array=array,
            points=points,
            padding=0
        )

        # floor(2.1) = 2, ceil(5.9) = 6
        # So should crop to [2:6, 2:6, 2:6]
        expected_shape = (4, 4, 4, 1)
        assert cropped.shape == expected_shape

    def test_single_point(self):
        """Test cropping with a single point."""
        array = np.arange(1000).reshape(10, 10, 10, 1)

        points = np.array([
            [5.0, 5.0, 5.0],
        ])

        cropped = get_cropped_region_from_array(
            array=array,
            points=points,
            padding=1
        )

        # floor(5) - 1 = 4, ceil(5) + 1 = 6
        # Should crop to [4:6, 4:6, 4:6]
        expected_shape = (2, 2, 2, 1)
        assert cropped.shape == expected_shape

    def test_array_values_preserved(self):
        """Test that the cropped array contains correct values."""
        # Create array with distinct values
        array = np.arange(8).reshape(2, 2, 2, 1)
        # array[0,0,0] = 0, array[0,0,1] = 1, etc.

        points = np.array([
            [0.0, 0.0, 0.0],
            [1.999, 1.999, 1.999],  # Just under 2 to stay in bounds
        ])

        cropped = get_cropped_region_from_array(
            array=array,
            points=points,
            padding=0
        )

        # floor(0) = 0, ceil(1.999) = 2, so crop [0:2, 0:2, 0:2]
        # Should return entire array
        np.testing.assert_array_equal(cropped, array)

    def test_edge_case_zero_padding(self):
        """Test edge case with zero padding and fractional point."""
        array = np.arange(27).reshape(3, 3, 3, 1)

        points = np.array([
            [1.2, 1.2, 1.2],
        ])

        cropped = get_cropped_region_from_array(
            array=array,
            points=points,
            padding=0
        )

        # floor(1.2) = 1, ceil(1.2) = 2
        # Should crop to [1:2, 1:2, 1:2]
        assert cropped.shape == (1, 1, 1, 1)
        assert cropped[0, 0, 0, 0] == array[1, 1, 1, 0]

    def test_asymmetric_bounds(self):
        """Test cropping with asymmetric point distribution."""
        array = np.arange(1000).reshape(10, 10, 10, 1)

        points = np.array([
            [2.0, 3.0, 4.0],
            [8.0, 5.0, 6.0],
        ])

        cropped = get_cropped_region_from_array(
            array=array,
            points=points,
            padding=1
        )

        # Dim 0: floor(2)-1=1 to ceil(8)+1=9 -> shape 8
        # Dim 1: floor(3)-1=2 to ceil(5)+1=6 -> shape 4
        # Dim 2: floor(4)-1=3 to ceil(6)+1=7 -> shape 4
        expected_shape = (8, 4, 4, 1)
        assert cropped.shape == expected_shape


class TestTransformPointsToTemplateAntsSpace:
    """Tests for the transform_points_to_template_ants_space function."""

    @pytest.fixture
    def sample_acquisition_axes(self):
        """Create sample acquisition axes."""
        return [
            AcquisitionAxis(
                dimension=0,
                direction=AcquisitionDirection.LEFT_TO_RIGHT,
                name=AcqusitionAxesName.X,
                unit="um",
                resolution=1.8
            ),
            AcquisitionAxis(
                dimension=1,
                direction=AcquisitionDirection.SUPERIOR_TO_INFERIOR,
                name=AcqusitionAxesName.Y,
                unit="um",
                resolution=1.8
            ),
            AcquisitionAxis(
                dimension=2,
                direction=AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
                name=AcqusitionAxesName.Z,
                unit="um",
                resolution=2.0
            )
        ]

    @pytest.fixture
    def sample_template_info(self):
        """Create sample template parameters."""
        return AntsImageParameters(
            orientation="RAS",
            dims=3,
            scale=(25.0, 25.0, 25.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )

    @pytest.fixture
    def sample_points(self):
        """Create sample points DataFrame."""
        return pd.DataFrame({
            'x': [10.0, 20.0, 30.0],
            'y': [15.0, 25.0, 35.0],
            'z': [5.0, 10.0, 15.0]
        })

    def test_points_reordered_by_dimension(
        self, sample_acquisition_axes, sample_template_info, sample_points
    ):
        """Test that points are reordered according to acquisition axes."""
        result = transform_points_to_template_ants_space(
            acquisition_axes=sample_acquisition_axes,
            ls_template_info=sample_template_info,
            points=sample_points,
            input_volume_shape=(100, 100, 100),
            registration_downsample=3.0
        )

        # Should return array with shape (n_points, 3)
        assert result.shape == (3, 3)
        assert isinstance(result, np.ndarray)

    def test_output_is_numpy_array(
        self, sample_acquisition_axes, sample_template_info, sample_points
    ):
        """Test that output is a numpy array."""
        result = transform_points_to_template_ants_space(
            acquisition_axes=sample_acquisition_axes,
            ls_template_info=sample_template_info,
            points=sample_points,
            input_volume_shape=(100, 100, 100),
            registration_downsample=3.0
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64 or result.dtype == np.float32


class TestApplyTransformsToPoints:
    """Tests for the apply_transforms_to_points function.

    Note: These tests require extensive mocking due to complex dependencies.
    Testing focuses on the crop_warp_to_bounding_box parameter behavior.
    """

    def test_crop_warp_parameter_controls_cropping(self, tmp_path):
        """Test that crop_warp_to_bounding_box parameter controls behavior."""
        # Create mock affine file
        affine_path = tmp_path / "affine.mat"
        affine_path.write_text("# Mock affine file")

        warp = np.zeros((50, 50, 50, 3))
        points = np.array([[25.0, 25.0, 25.0]])

        template_params = AntsImageParameters(
            orientation="RAS",
            dims=3,
            scale=(25.0, 25.0, 25.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.array([1.0, 1.0, 1.0])  # Simplified for mocking
        )

        with patch(
            'aind_smartspim_transform_utils.utils.utils.apply_transforms_to_points'
        ) as mock_affine, \
        patch(
            'aind_smartspim_transform_utils.utils.utils.convert_from_ants_space'
        ) as mock_convert, \
        patch(
            'deep_ccf_registration.utils.interpolation.interpolate'
        ) as mock_interp, \
        patch(
            'deep_ccf_registration.utils.transforms.get_cropped_region_from_array'
        ) as mock_crop:

            # Setup mocks
            mock_affine.return_value = points.copy()
            mock_convert.return_value = points.copy()
            mock_interp.return_value = Mock(
                squeeze=lambda: Mock(
                    T=Mock(numpy=lambda: np.zeros((1, 3)))
                )
            )
            mock_crop.return_value = warp

            # Test with crop_warp_to_bounding_box=True
            apply_transforms_to_points(
                points=points,
                affine_path=affine_path,
                warp=warp,
                template_parameters=template_params,
                crop_warp_to_bounding_box=True
            )
            mock_crop.assert_called_once()

            mock_crop.reset_mock()

            # Test with crop_warp_to_bounding_box=False
            apply_transforms_to_points(
                points=points,
                affine_path=affine_path,
                warp=warp,
                template_parameters=template_params,
                crop_warp_to_bounding_box=False
            )
            mock_crop.assert_not_called()


class TestMapPointsToRightHemisphere:
    @pytest.fixture
    def template_parameters(self):
        return AntsImageParameters(
            orientation="RAS",
            dims=3,
            scale=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.array([1.0, 1.0, 1.0])
        )

    def test_points_in_right_hemisphere_unchanged(self, template_parameters):
        template_points = np.array([
            [[25.0, 10.0, 5.0], [55.0, 10.0, 5.0]],
            [[35.0, 15.0, 10.0], [45.0, 15.0, 10.0]],
        ])
        original = template_points.copy()

        result = map_points_to_right_hemisphere(
            template_points=template_points,
            template_parameters=template_parameters,
            ml_dim_size=100
        )

        np.testing.assert_array_equal(result, original)
        np.testing.assert_array_equal(template_points, original)

    def test_points_mirrored_when_all_left(self, template_parameters):
        template_points = np.array([
            [[80.0, 5.0, 5.0], [90.0, 5.0, 5.0]],
            [[85.0, 10.0, 5.0], [95.0, 10.0, 5.0]],
        ])

        result = map_points_to_right_hemisphere(
            template_points=template_points,
            template_parameters=template_parameters,
            ml_dim_size=100
        )

        expected = template_points.copy()
        expected[:, :, 0] = 99 - expected[:, :, 0]
        np.testing.assert_array_equal(result, expected)
