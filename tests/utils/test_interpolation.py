import numpy as np
import torch
import pytest

from deep_ccf_registration.utils.interpolation import _prepare_grid_sample, interpolate


class TestPrepareGridSample:
    """Tests for the _prepare_grid_sample function."""

    def test_4d_array_conversion(self):
        """Test conversion of 4D array (with channel dimension)."""
        # Create a 4D array (D, H, W, C) - e.g., displacement field
        array = np.random.rand(10, 20, 30, 3).astype(np.float32)
        voxels = np.array([
            [5.0, 10.0, 15.0],
            [7.0, 12.0, 18.0],
        ])

        array_tensor, voxels_tensor = _prepare_grid_sample(array, voxels)

        # Array should be (1, 3, 10, 20, 30) - batch, channels, D, H, W
        assert array_tensor.shape == (1, 3, 10, 20, 30)
        assert isinstance(array_tensor, torch.Tensor)

    def test_3d_array_conversion(self):
        """Test conversion of 3D array (no channel dimension)."""
        # Create a 3D array (D, H, W)
        array = np.random.rand(10, 20, 30).astype(np.float32)
        voxels = np.array([
            [5.0, 10.0, 15.0],
        ])

        array_tensor, voxels_tensor = _prepare_grid_sample(array, voxels)

        # Array should be (1, 1, 10, 20, 30) - batch, channels, D, H, W
        assert array_tensor.shape == (1, 1, 10, 20, 30)
        assert isinstance(array_tensor, torch.Tensor)

    def test_voxels_normalized_to_minus_one_to_one(self):
        """Test that voxel coordinates are normalized to [-1, 1] range."""
        array = np.zeros((10, 20, 30, 3), dtype=np.float32)

        # Test corner voxels
        voxels = np.array([
            [0.0, 0.0, 0.0],      # Should map to -1, -1, -1
            [9.0, 19.0, 29.0],    # Should map to 1, 1, 1
            [4.5, 9.5, 14.5],     # Should map to 0, 0, 0 (center)
        ])

        _, voxels_tensor = _prepare_grid_sample(array, voxels)

        # Extract normalized coordinates (1, N, 1, 1, 3) -> (N, 3)
        normalized = voxels_tensor.squeeze().numpy()

        # Check corner mapping (note: coordinates are reordered to W, H, D)
        # Original [D, H, W] -> normalized [W, H, D]

        # First point [0, 0, 0] -> normalized should be [-1, -1, -1] in W, H, D order
        np.testing.assert_allclose(normalized[0], [-1.0, -1.0, -1.0], atol=1e-6)

        # Last point [9, 19, 29] -> normalized should be [1, 1, 1] in W, H, D order
        np.testing.assert_allclose(normalized[1], [1.0, 1.0, 1.0], atol=1e-6)

        # Center point should be close to [0, 0, 0]
        np.testing.assert_allclose(normalized[2], [0.0, 0.0, 0.0], atol=1e-6)

    def test_voxels_reordered_from_dhw_to_whd(self):
        """Test that voxel coordinates are reordered from (D,H,W) to (W,H,D)."""
        array = np.zeros((10, 20, 30, 3), dtype=np.float32)

        # Create a voxel with distinct coordinates in each dimension
        voxels = np.array([
            [9.0, 19.0, 29.0],  # D=9, H=19, W=29 (max values)
        ])

        _, voxels_tensor = _prepare_grid_sample(array, voxels)

        # Extract normalized coordinates
        normalized = voxels_tensor.squeeze().numpy()

        # All should be 1.0 (max normalized value)
        # Order should be [W, H, D] = [1, 1, 1]
        assert normalized.shape == (3,)
        np.testing.assert_allclose(normalized, [1.0, 1.0, 1.0], atol=1e-6)

    def test_voxels_tensor_shape(self):
        """Test that voxels tensor has correct shape for grid_sample."""
        array = np.zeros((10, 20, 30, 3), dtype=np.float32)
        voxels = np.array([
            [5.0, 10.0, 15.0],
            [7.0, 12.0, 18.0],
            [3.0, 8.0, 20.0],
        ])

        _, voxels_tensor = _prepare_grid_sample(array, voxels)

        # Should be (1, N, 1, 1, 3) where N is number of points
        assert voxels_tensor.shape == (1, 3, 1, 1, 3)
        assert isinstance(voxels_tensor, torch.Tensor)

    def test_dtype_preservation_float32(self):
        """Test that float32 dtype is preserved."""
        array = np.zeros((10, 20, 30, 3), dtype=np.float32)
        voxels = np.array([[5.0, 10.0, 15.0]], dtype=np.float64)

        array_tensor, voxels_tensor = _prepare_grid_sample(array, voxels)

        assert array_tensor.dtype == torch.float32
        assert voxels_tensor.dtype == torch.float32

    def test_dtype_preservation_float16(self):
        """Test that float16 dtype is preserved."""
        array = np.zeros((10, 20, 30, 3), dtype=np.float16)
        voxels = np.array([[5.0, 10.0, 15.0]], dtype=np.float64)

        array_tensor, voxels_tensor = _prepare_grid_sample(array, voxels)

        assert array_tensor.dtype == torch.float16
        assert voxels_tensor.dtype == torch.float16

    def test_single_voxel(self):
        """Test handling of a single voxel."""
        array = np.zeros((10, 20, 30, 3), dtype=np.float32)
        voxels = np.array([[5.0, 10.0, 15.0]])

        array_tensor, voxels_tensor = _prepare_grid_sample(array, voxels)

        assert array_tensor.shape == (1, 3, 10, 20, 30)
        assert voxels_tensor.shape == (1, 1, 1, 1, 3)

    def test_many_voxels(self):
        """Test handling of many voxels."""
        array = np.zeros((10, 20, 30, 3), dtype=np.float32)
        n_voxels = 100
        voxels = np.random.rand(n_voxels, 3) * np.array([9, 19, 29])

        array_tensor, voxels_tensor = _prepare_grid_sample(array, voxels)

        assert array_tensor.shape == (1, 3, 10, 20, 30)
        assert voxels_tensor.shape == (1, n_voxels, 1, 1, 3)

    def test_invalid_array_shape_raises_error(self):
        """Test that invalid array shapes raise ValueError."""
        # 2D array should raise error
        array = np.zeros((10, 20), dtype=np.float32)
        voxels = np.array([[5.0, 10.0, 15.0]])

        with pytest.raises(ValueError, match="cannot handle array of shape"):
            _prepare_grid_sample(array, voxels)

    def test_normalization_formula(self):
        """Test the normalization formula: 2.0 * voxels / (shape - 1) - 1.0"""
        array = np.zeros((11, 21, 31, 3), dtype=np.float32)  # Shape - 1 = [10, 20, 30]

        # Test specific points
        voxels = np.array([
            [0.0, 0.0, 0.0],      # Min: 2*0/10 - 1 = -1
            [5.0, 10.0, 15.0],    # Mid: 2*5/10 - 1 = 0, etc.
            [10.0, 20.0, 30.0],   # Max: 2*10/10 - 1 = 1
        ])

        _, voxels_tensor = _prepare_grid_sample(array, voxels)
        normalized = voxels_tensor.squeeze().numpy()

        # After reordering [D, H, W] -> [W, H, D]:
        # Point 0: [0, 0, 0] -> [-1, -1, -1] in W,H,D
        # Point 1: [5, 10, 15] -> [0, 0, 0] in W,H,D
        # Point 2: [10, 20, 30] -> [1, 1, 1] in W,H,D

        np.testing.assert_allclose(normalized[0], [-1.0, -1.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(normalized[1], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(normalized[2], [1.0, 1.0, 1.0], atol=1e-6)

    def test_array_values_preserved(self):
        """Test that array values are preserved during conversion."""
        # Create array with specific values
        array = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ], dtype=np.float32)  # Shape (2, 2, 3, 3) after adding channel dim

        array_4d = np.stack([array, array * 2, array * 3], axis=-1)  # (2, 2, 3, 3)
        voxels = np.array([[0.0, 0.0, 0.0]])

        array_tensor, _ = _prepare_grid_sample(array_4d, voxels)

        # Check that values are preserved (just reordered)
        # Original: (D=2, H=2, W=3, C=3)
        # Result: (1, C=3, D=2, H=2, W=3)
        assert array_tensor[0, 0, 0, 0, 0] == 1  # First channel, first element
        assert array_tensor[0, 1, 0, 0, 0] == 2  # Second channel, first element
        assert array_tensor[0, 2, 0, 0, 0] == 3  # Third channel, first element

    def test_3d_array_channel_addition(self):
        """Test that 3D array gets a channel dimension added."""
        array = np.random.rand(5, 6, 7).astype(np.float32)
        voxels = np.array([[2.0, 3.0, 4.0]])

        array_tensor, _ = _prepare_grid_sample(array, voxels)

        # Should add channel dimension: (5, 6, 7) -> (1, 1, 5, 6, 7)
        assert array_tensor.shape == (1, 1, 5, 6, 7)
        assert array_tensor.dim() == 5


class TestInterpolate:
    """Integration tests for the interpolate function."""

    def test_interpolate_basic(self):
        """Test basic interpolation functionality."""
        # Create a simple array
        array = np.zeros((10, 10, 10, 3), dtype=np.float32)
        array[5, 5, 5, :] = [1.0, 2.0, 3.0]  # Set center point

        # Sample at the center
        voxels = np.array([[5.0, 5.0, 5.0]])

        result = interpolate(array, voxels, mode='bilinear')

        # Result should be close to [1, 2, 3]
        result_np = result.squeeze().numpy()
        assert result_np.shape == (3,)
        np.testing.assert_allclose(result_np, [1.0, 2.0, 3.0], atol=0.01)

    def test_interpolate_returns_tensor(self):
        """Test that interpolate returns a torch tensor."""
        array = np.zeros((10, 10, 10, 3), dtype=np.float32)
        voxels = np.array([[5.0, 5.0, 5.0]])

        result = interpolate(array, voxels, mode='bilinear')

        assert isinstance(result, torch.Tensor)

    def test_interpolate_output_shape(self):
        """Test that interpolate output has correct shape."""
        array = np.zeros((10, 10, 10, 3), dtype=np.float32)
        voxels = np.array([
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
        ])

        result = interpolate(array, voxels, mode='bilinear')

        # Output should be (1, 3, 2, 1, 1) - batch, channels, N_points, 1, 1
        assert result.shape == (1, 3, 2, 1, 1)
