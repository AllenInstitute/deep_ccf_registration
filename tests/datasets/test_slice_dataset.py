import numpy as np
import pandas as pd
import pytest
from aind_smartspim_transform_utils.utils.utils import AcquisitionDirection, AcqusitionAxesName
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from deep_ccf_registration.datasets.slice_dataset import (
    SliceDataset,
    TissueBoundingBox,
    TissueBoundingBoxes,
    Patch,
    TrainMode,
    _create_coordinate_dataframe,
    _calculate_non_pad_mask,
)
from deep_ccf_registration.metadata import (
    AcquisitionAxis,
    SliceOrientation,
    SubjectMetadata,
)


class TestNormalizeOrientation:
    """Tests for the _normalize_orientation method of SliceDataset."""

    @pytest.fixture
    def dataset_instance(self):
        """
        Create a minimal SliceDataset instance for testing.

        We only need to set the normalize_orientation_map since we're testing
        the _normalize_orientation method directly.
        """
        dataset = object.__new__(SliceDataset)
        dataset._normalize_orientation_map = {
            SliceOrientation.SAGITTAL: [
                AcquisitionDirection.SUPERIOR_TO_INFERIOR,
                AcquisitionDirection.ANTERIOR_TO_POSTERIOR
            ]
        }
        return dataset

    # Test cases: (slice_axis_dim, directions, expected_shape)
    # Each tuple: (slice_axis_dimension, [dim0, dim1, dim2], expected_shape)
    @pytest.mark.parametrize("slice_axis_dim,directions,expected_shape", [
        # LSA - already normalized (Left-Superior-Anterior)
        (0, [AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.SUPERIOR_TO_INFERIOR,
             AcquisitionDirection.ANTERIOR_TO_POSTERIOR], (3, 4)),

        # LSP - flip horizontal (Left-Superior-Posterior)
        (0, [AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.SUPERIOR_TO_INFERIOR,
             AcquisitionDirection.POSTERIOR_TO_ANTERIOR], (3, 4)),

        # LIA - flip vertical (Left-Inferior-Anterior)
        (0, [AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.INFERIOR_TO_SUPERIOR,
             AcquisitionDirection.ANTERIOR_TO_POSTERIOR], (3, 4)),

        # LIP - flip both (Left-Inferior-Posterior)
        (0, [AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.INFERIOR_TO_SUPERIOR,
             AcquisitionDirection.POSTERIOR_TO_ANTERIOR], (3, 4)),

        # LAS - transpose (Left-Anterior-Superior)
        (0, [AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
             AcquisitionDirection.SUPERIOR_TO_INFERIOR], (4, 3)),

        # LAI - transpose + flip (Left-Anterior-Inferior)
        (0, [AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
             AcquisitionDirection.INFERIOR_TO_SUPERIOR], (4, 3)),

        # LPS - transpose + flip (Left-Posterior-Superior)
        (0, [AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.POSTERIOR_TO_ANTERIOR,
             AcquisitionDirection.SUPERIOR_TO_INFERIOR], (4, 3)),

        # LPI - transpose + flip both (Left-Posterior-Inferior)
        (0, [AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.POSTERIOR_TO_ANTERIOR,
             AcquisitionDirection.INFERIOR_TO_SUPERIOR], (4, 3)),

        # RSA - same as LSA (Right vs Left doesn't affect 2D slice)
        (0, [AcquisitionDirection.RIGHT_TO_LEFT,
             AcquisitionDirection.SUPERIOR_TO_INFERIOR,
             AcquisitionDirection.ANTERIOR_TO_POSTERIOR], (3, 4)),

        # SAL - slice axis on different dimension
        (2, [AcquisitionDirection.SUPERIOR_TO_INFERIOR,
             AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
             AcquisitionDirection.LEFT_TO_RIGHT], (3, 4)),

        # IAL - flip vertical, slice axis on dim 2
        (2, [AcquisitionDirection.INFERIOR_TO_SUPERIOR,
             AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
             AcquisitionDirection.LEFT_TO_RIGHT], (3, 4)),

        # SPL - flip horizontal, slice axis on dim 2
        (2, [AcquisitionDirection.SUPERIOR_TO_INFERIOR,
             AcquisitionDirection.POSTERIOR_TO_ANTERIOR,
             AcquisitionDirection.LEFT_TO_RIGHT], (3, 4)),

        # ASL - transpose, slice axis on dim 2
        (2, [AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
             AcquisitionDirection.SUPERIOR_TO_INFERIOR,
             AcquisitionDirection.LEFT_TO_RIGHT], (4, 3)),

        # ALS - different permutation
        (1, [AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
             AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.SUPERIOR_TO_INFERIOR], (4, 3)),

        # SLA - different permutation
        (1, [AcquisitionDirection.SUPERIOR_TO_INFERIOR,
             AcquisitionDirection.LEFT_TO_RIGHT,
             AcquisitionDirection.ANTERIOR_TO_POSTERIOR], (3, 4)),
    ])
    def test_orientation_normalization(
        self, dataset_instance, slice_axis_dim, directions, expected_shape
    ):
        """
        Test that various orientations are correctly normalized to SA.

        This parameterized test covers multiple orientation combinations to verify
        that all are properly normalized to Superior-Anterior orientation.
        """
        # Create a 3x4 test slice with identifiable pattern
        slice_image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ], dtype=np.float32)

        # Create template points
        template_points = np.zeros((3, 4, 3), dtype=np.float32)
        for i in range(3):
            for j in range(4):
                template_points[i, j] = [i * 10, j * 10, i + j]

        # Build acquisition axes
        acquisition_axes = [
            AcquisitionAxis(
                dimension=i,
                direction=directions[i],
                name=[AcqusitionAxesName.X, AcqusitionAxesName.Y, AcqusitionAxesName.Z][i],
                unit="um",
                resolution=1.0
            )
            for i in range(3)
        ]

        slice_axis = acquisition_axes[slice_axis_dim]

        # Apply normalization
        normalized_slice, normalized_points = dataset_instance._normalize_orientation(
            slice=slice_image.copy(),
            template_points=template_points.copy(),
            acquisition_axes=acquisition_axes,
            orientation=SliceOrientation.SAGITTAL,
            slice_axis=slice_axis
        )

        # Verify output shape matches expected
        assert normalized_slice.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {normalized_slice.shape}"

        # Verify template points shape matches
        assert normalized_points.shape == (*expected_shape, 3), \
            f"Expected points shape {(*expected_shape, 3)}, got {normalized_points.shape}"

        # Verify all values are preserved (just rearranged)
        original_values = sorted(slice_image.flatten())
        normalized_values = sorted(normalized_slice.flatten())
        np.testing.assert_array_equal(normalized_values, original_values)

        # Verify template points values are preserved
        original_coords = sorted([tuple(template_points[i, j])
                                  for i in range(template_points.shape[0])
                                  for j in range(template_points.shape[1])])
        normalized_coords = sorted([tuple(normalized_points[i, j])
                                    for i in range(normalized_points.shape[0])
                                    for j in range(normalized_points.shape[1])])
        assert original_coords == normalized_coords

    def test_slice_and_points_transform_together(self, dataset_instance):
        """
        Verify that slice and template points undergo identical spatial transformations.

        This ensures that the correspondence between image pixels and their template
        coordinates is preserved during normalization.
        """
        # Create a 2x2 slice with unique values at each position
        slice_image = np.array([
            [100, 200],
            [300, 400]
        ], dtype=np.float32)

        # Create template points with unique coordinates at each position
        template_points = np.array([
            [[10, 20, 30], [11, 21, 31]],
            [[12, 22, 32], [13, 23, 33]]
        ], dtype=np.float32)

        # Use LPI orientation (will be flipped and transposed to SA)
        acquisition_axes = [
            AcquisitionAxis(
                dimension=0,
                direction=AcquisitionDirection.LEFT_TO_RIGHT,
                name=AcqusitionAxesName.X,
                unit="um",
                resolution=1.0
            ),
            AcquisitionAxis(
                dimension=1,
                direction=AcquisitionDirection.POSTERIOR_TO_ANTERIOR,
                name=AcqusitionAxesName.Y,
                unit="um",
                resolution=1.0
            ),
            AcquisitionAxis(
                dimension=2,
                direction=AcquisitionDirection.INFERIOR_TO_SUPERIOR,
                name=AcqusitionAxesName.Z,
                unit="um",
                resolution=1.0
            )
        ]

        slice_axis = acquisition_axes[0]

        # Apply normalization
        normalized_slice, normalized_points = dataset_instance._normalize_orientation(
            slice=slice_image.copy(),
            template_points=template_points.copy(),
            acquisition_axes=acquisition_axes,
            orientation=SliceOrientation.SAGITTAL,
            slice_axis=slice_axis
        )

        # For LPI->LSA transformation: flipud, fliplr, transpose
        # Verify the transformations moved values and coordinates together

        # Original position [0,0] (value=100, coords=[10,20,30])
        # After flipud+fliplr+transpose -> position [1,1]
        assert normalized_slice[1, 1] == 100
        np.testing.assert_array_equal(normalized_points[1, 1], [10, 20, 30])

        # Original position [0,1] (value=200, coords=[11,21,31])
        # After flipud+fliplr+transpose -> position [0,1]
        assert normalized_slice[0, 1] == 200
        np.testing.assert_array_equal(normalized_points[0, 1], [11, 21, 31])

        # Original position [1,0] (value=300, coords=[12,22,32])
        # After flipud+fliplr+transpose -> position [1,0]
        assert normalized_slice[1, 0] == 300
        np.testing.assert_array_equal(normalized_points[1, 0], [12, 22, 32])

        # Original position [1,1] (value=400, coords=[13,23,33])
        # After flipud+fliplr+transpose -> position [0,0]
        assert normalized_slice[0, 0] == 400
        np.testing.assert_array_equal(normalized_points[0, 0], [13, 23, 33])
