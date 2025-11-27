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


class TestCreateCoordinateDataframe:
    """Tests for the _create_coordinate_dataframe function."""

    def test_basic_coordinate_creation(self):
        """Test basic coordinate dataframe creation."""
        axes = [
            AcquisitionAxis(
                dimension=0,
                direction=AcquisitionDirection.LEFT_TO_RIGHT,
                name=AcqusitionAxesName.X,
                unit="um",
                resolution=1.0
            ),
            AcquisitionAxis(
                dimension=1,
                direction=AcquisitionDirection.SUPERIOR_TO_INFERIOR,
                name=AcqusitionAxesName.Y,
                unit="um",
                resolution=1.0
            ),
            AcquisitionAxis(
                dimension=2,
                direction=AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
                name=AcqusitionAxesName.Z,
                unit="um",
                resolution=1.0
            )
        ]
        slice_axis = axes[0]

        df = _create_coordinate_dataframe(
            patch_height=2,
            patch_width=3,
            start_x=10,
            start_y=20,
            fixed_index_value=5,
            slice_axis=slice_axis,
            axes=axes
        )

        # Verify shape
        assert len(df) == 6  # 2 * 3 = 6 points
        assert list(df.columns) == ['x', 'y', 'z']

        # Verify slice dimension is fixed
        assert all(df['x'] == 5)

        # Verify y coordinates (height dimension)
        assert sorted(df['y'].unique()) == [20, 21]

        # Verify z coordinates (width dimension)
        assert sorted(df['z'].unique()) == [10, 11, 12]

    def test_coordinate_ordering(self):
        """Test that coordinates are ordered correctly (row-major)."""
        axes = [
            AcquisitionAxis(
                dimension=0,
                direction=AcquisitionDirection.LEFT_TO_RIGHT,
                name=AcqusitionAxesName.X,
                unit="um",
                resolution=1.0
            ),
            AcquisitionAxis(
                dimension=1,
                direction=AcquisitionDirection.SUPERIOR_TO_INFERIOR,
                name=AcqusitionAxesName.Y,
                unit="um",
                resolution=1.0
            ),
            AcquisitionAxis(
                dimension=2,
                direction=AcquisitionDirection.ANTERIOR_TO_POSTERIOR,
                name=AcqusitionAxesName.Z,
                unit="um",
                resolution=1.0
            )
        ]
        slice_axis = axes[0]

        df = _create_coordinate_dataframe(
            patch_height=2,
            patch_width=2,
            start_x=0,
            start_y=0,
            fixed_index_value=0,
            slice_axis=slice_axis,
            axes=axes
        )

        # With indexing='ij', coordinates should be:
        # (y=0, z=0), (y=0, z=1), (y=1, z=0), (y=1, z=1)
        expected_coords = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1)
        ]

        for i, (x, y, z) in enumerate(expected_coords):
            assert df.iloc[i]['x'] == x
            assert df.iloc[i]['y'] == y
            assert df.iloc[i]['z'] == z


class TestCalculateNonPadMask:
    """Tests for the _calculate_non_pad_mask function."""

    def test_no_padding(self):
        """Test mask creation with no padding."""
        mask = _calculate_non_pad_mask(shape=(10, 20), pad_transform={})

        assert mask.shape == (10, 20)
        assert mask.dtype == np.uint8
        assert np.all(mask == 1)

    def test_with_padding(self):
        """Test mask creation with padding."""
        pad_transform = {
            'pad_top': 2,
            'pad_left': 3,
            'shape': (5, 8)  # Original image size before padding
        }

        mask = _calculate_non_pad_mask(shape=(10, 15), pad_transform=pad_transform)

        assert mask.shape == (10, 15)

        # Check padded region is 0
        assert np.all(mask[:2, :] == 0)  # Top padding
        assert np.all(mask[:, :3] == 0)  # Left padding

        # Check non-padded region is 1
        # Original image is at [2:7, 3:11]
        assert np.all(mask[2:7, 3:11] == 1)

    def test_padding_values(self):
        """Test specific padding boundaries."""
        pad_transform = {
            'pad_top': 1,
            'pad_left': 1,
            'shape': (3, 3)
        }

        mask = _calculate_non_pad_mask(shape=(5, 5), pad_transform=pad_transform)

        # Non-pad region should be [1:4, 1:4]
        expected_mask = np.zeros((5, 5), dtype='uint8')
        expected_mask[1:4, 1:4] = 1

        np.testing.assert_array_equal(mask, expected_mask)


class TestTissueBoundingBox:
    """Tests for the TissueBoundingBox pydantic model."""

    def test_creation(self):
        """Test basic creation of TissueBoundingBox."""
        bbox = TissueBoundingBox(y=10, x=20, width=100, height=50)

        assert bbox.y == 10
        assert bbox.x == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_validation(self):
        """Test that pydantic validates types."""
        # Valid
        bbox = TissueBoundingBox(y=10, x=20, width=100, height=50)
        assert isinstance(bbox, TissueBoundingBox)

        # Test with whole number floats (should be coerced to int)
        bbox = TissueBoundingBox(y=10.0, x=20.0, width=100.0, height=50.0)
        assert bbox.y == 10
        assert bbox.x == 20

        # Test that fractional floats raise validation error
        with pytest.raises(Exception):  # pydantic.ValidationError
            TissueBoundingBox(y=10.5, x=20, width=100, height=50)


class TestPatch:
    """Tests for the Patch dataclass."""

    def test_creation(self):
        """Test creating a Patch instance."""
        patch = Patch(
            dataset_idx=0,
            slice_idx=5,
            x=100,
            y=200,
            orientation=SliceOrientation.SAGITTAL
        )

        assert patch.dataset_idx == 0
        assert patch.slice_idx == 5
        assert patch.x == 100
        assert patch.y == 200
        assert patch.orientation == SliceOrientation.SAGITTAL

    def test_equality(self):
        """Test that patches can be compared for equality."""
        patch1 = Patch(
            dataset_idx=0,
            slice_idx=5,
            x=100,
            y=200,
            orientation=SliceOrientation.SAGITTAL
        )
        patch2 = Patch(
            dataset_idx=0,
            slice_idx=5,
            x=100,
            y=200,
            orientation=SliceOrientation.SAGITTAL
        )
        patch3 = Patch(
            dataset_idx=1,
            slice_idx=5,
            x=100,
            y=200,
            orientation=SliceOrientation.SAGITTAL
        )

        assert patch1 == patch2
        assert patch1 != patch3


class TestGetPatchPositions:
    """Tests for the _get_patch_positions method of SliceDataset."""

    def test_no_patch_size(self):
        """Test when patch_size is None."""
        dataset = object.__new__(SliceDataset)
        dataset._patch_size = None

        bbox = TissueBoundingBox(y=10, x=20, width=100, height=50)
        positions = dataset._get_patch_positions(bbox)

        # Should return single position at bounding box start
        assert positions == [(10, 20)]

    def test_single_patch_fits_exactly(self):
        """Test when bounding box exactly fits one patch."""
        dataset = object.__new__(SliceDataset)
        dataset._patch_size = (50, 100)

        bbox = TissueBoundingBox(y=10, x=20, width=100, height=50)
        positions = dataset._get_patch_positions(bbox)

        # Should return single position
        assert len(positions) == 1
        assert positions[0] == (10, 20)

    def test_multiple_patches_no_overlap(self):
        """Test tiling multiple patches with no overlap."""
        dataset = object.__new__(SliceDataset)
        dataset._patch_size = (25, 50)

        bbox = TissueBoundingBox(y=0, x=0, width=100, height=50)
        positions = dataset._get_patch_positions(bbox)

        # Should tile 2x2 = 4 patches
        expected = [
            (0, 0), (0, 50),
            (25, 0), (25, 50)
        ]
        assert positions == expected

    def test_patch_larger_than_bbox(self):
        """Test when patch is larger than bounding box."""
        dataset = object.__new__(SliceDataset)
        dataset._patch_size = (100, 200)

        bbox = TissueBoundingBox(y=10, x=20, width=50, height=30)
        positions = dataset._get_patch_positions(bbox)

        # Should return single position at bounding box start
        assert len(positions) == 1
        assert positions[0] == (10, 20)

    def test_edge_adjustment(self):
        """Test that patches are adjusted at edges."""
        dataset = object.__new__(SliceDataset)
        dataset._patch_size = (30, 30)

        bbox = TissueBoundingBox(y=0, x=0, width=50, height=50)
        positions = dataset._get_patch_positions(bbox)

        # Should have patches at (0,0), (0,20), (20,0), (20,20)
        # Last patches adjusted to fit within bbox
        assert (0, 0) in positions
        assert (0, 20) in positions  # width=50, 50-30=20
        assert (20, 0) in positions  # height=50, 50-30=20
        assert (20, 20) in positions


class TestGetSliceFromIdx:
    """Tests for the _get_slice_from_idx method of SliceDataset."""

    def test_single_subject(self):
        """Test with a single subject."""
        dataset = object.__new__(SliceDataset)
        dataset._slice_ranges = {
            SliceOrientation.SAGITTAL: [[0, 1, 2, 3, 4]]
        }

        dataset_idx, slice_idx = dataset._get_slice_from_idx(
            idx=2,
            orientation=SliceOrientation.SAGITTAL
        )

        assert dataset_idx == 0
        assert slice_idx == 2

    def test_multiple_subjects(self):
        """Test with multiple subjects."""
        dataset = object.__new__(SliceDataset)
        dataset._slice_ranges = {
            SliceOrientation.SAGITTAL: [
                [0, 1, 2],      # Subject 0: 3 slices
                [0, 1, 2, 3],   # Subject 1: 4 slices
                [0, 1]          # Subject 2: 2 slices
            ]
        }

        # Test first subject
        dataset_idx, slice_idx = dataset._get_slice_from_idx(
            idx=0,
            orientation=SliceOrientation.SAGITTAL
        )
        assert dataset_idx == 0
        assert slice_idx == 0

        # Test last slice of first subject
        dataset_idx, slice_idx = dataset._get_slice_from_idx(
            idx=2,
            orientation=SliceOrientation.SAGITTAL
        )
        assert dataset_idx == 0
        assert slice_idx == 2

        # Test first slice of second subject (global idx=3)
        dataset_idx, slice_idx = dataset._get_slice_from_idx(
            idx=3,
            orientation=SliceOrientation.SAGITTAL
        )
        assert dataset_idx == 1
        assert slice_idx == 0

        # Test third subject
        dataset_idx, slice_idx = dataset._get_slice_from_idx(
            idx=7,
            orientation=SliceOrientation.SAGITTAL
        )
        assert dataset_idx == 2
        assert slice_idx == 0

    def test_non_contiguous_slice_indices(self):
        """Test with non-contiguous slice indices."""
        dataset = object.__new__(SliceDataset)
        dataset._slice_ranges = {
            SliceOrientation.SAGITTAL: [
                [5, 10, 15],  # Subject 0: slices 5, 10, 15
            ]
        }

        # Global index 1 should map to slice 10
        dataset_idx, slice_idx = dataset._get_slice_from_idx(
            idx=1,
            orientation=SliceOrientation.SAGITTAL
        )
        assert dataset_idx == 0
        assert slice_idx == 10


class TestGetNumSlicesInAxisPerSubject:
    """Tests for the _get_num_slices_in_axis_per_subject method."""

    def test_counts(self):
        """Test that correct counts are returned."""
        dataset = object.__new__(SliceDataset)
        dataset._slice_ranges = {
            SliceOrientation.SAGITTAL: [
                [0, 1, 2],
                [0, 1, 2, 3, 4],
                [0]
            ]
        }

        counts = dataset._get_num_slices_in_axis_per_subject(
            orientation=SliceOrientation.SAGITTAL
        )

        assert counts == [3, 5, 1]


class TestDatasetLen:
    """Tests for the __len__ method of SliceDataset."""

    def test_train_mode_single_orientation(self):
        """Test length in train mode with single orientation."""
        dataset = object.__new__(SliceDataset)
        dataset._mode = TrainMode.TRAIN
        dataset._patch_size = (256, 256)
        dataset._orientation = [SliceOrientation.SAGITTAL]
        dataset._slice_ranges = {
            SliceOrientation.SAGITTAL: [
                [0, 1, 2],
                [0, 1, 2, 3, 4]
            ]
        }

        assert len(dataset) == 8  # 3 + 5

    def test_train_mode_multiple_orientations(self):
        """Test length in train mode with multiple orientations."""
        dataset = object.__new__(SliceDataset)
        dataset._mode = TrainMode.TRAIN
        dataset._patch_size = (256, 256)
        dataset._orientation = [
            SliceOrientation.SAGITTAL,
            SliceOrientation.CORONAL
        ]
        dataset._slice_ranges = {
            SliceOrientation.SAGITTAL: [[0, 1, 2]],
            SliceOrientation.CORONAL: [[0, 1]]
        }

        assert len(dataset) == 5  # 3 + 2

    def test_test_mode_with_patches(self):
        """Test length in test mode with precomputed patches."""
        dataset = object.__new__(SliceDataset)
        dataset._mode = TrainMode.TEST
        dataset._patch_size = (256, 256)
        dataset._precomputed_patches = [
            Patch(0, 0, 0, 0, SliceOrientation.SAGITTAL),
            Patch(0, 0, 256, 0, SliceOrientation.SAGITTAL),
            Patch(0, 0, 0, 256, SliceOrientation.SAGITTAL),
        ]

        assert len(dataset) == 3

    def test_test_mode_no_patch_size(self):
        """Test length in test mode without patch size."""
        dataset = object.__new__(SliceDataset)
        dataset._mode = TrainMode.TEST
        dataset._patch_size = None
        dataset._orientation = [SliceOrientation.SAGITTAL]
        dataset._slice_ranges = {
            SliceOrientation.SAGITTAL: [[0, 1, 2]]
        }

        # Should use slice count, not patch count
        assert len(dataset) == 3
