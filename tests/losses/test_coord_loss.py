import numpy as np
import torch
import pytest
from aind_smartspim_transform_utils.io.file_io import AntsImageParameters

from deep_ccf_registration.losses.coord_loss import (
    mirror_points,
    HemisphereAgnosticCoordLoss,
)
from deep_ccf_registration.metadata import SliceOrientation


class TestMirrorPoints:
    """Tests for the mirror_points function."""

    @pytest.fixture
    def simple_template_params(self):
        """Create simple template parameters for testing."""
        return AntsImageParameters(
            orientation="RAS",
            dims=3,
            scale=(1.0, 1.0, 1.0),  # 1mm isotropic
            origin=(0.0, 0.0, 0.0),
            direction=np.array([1.0, 1.0, 1.0])
        )

    @pytest.fixture
    def complex_template_params(self):
        """Create template parameters with non-unit scaling and offset."""
        return AntsImageParameters(
            orientation="RAS",
            dims=3,
            scale=(0.025, 0.025, 0.025),  # 25 micron spacing
            origin=(-6.6, -10.2, -7.9),
            direction=np.array([1.0, 1.0, 1.0])
        )

    def test_center_point_unchanged_identity_transform(self, simple_template_params):
        """Test that center point in ML dimension stays at center after mirroring."""
        ml_dim_size = 100  # 0 to 99 in index space
        center_ml = (ml_dim_size - 1) / 2  # 49.5

        # Create coordinate field with shape (B=1, C=3, H=10, W=10)
        # All pixels have ML=center_ml, AP=50.0, DV=50.0
        pred = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
        pred[0, 0, :, :] = center_ml  # ML channel
        pred[0, 1, :, :] = 50.0        # AP channel
        pred[0, 2, :, :] = 50.0        # DV channel

        flipped = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # Center should remain at center for all pixels
        assert torch.allclose(flipped[0, 0], pred[0, 0], atol=1e-5)

    def test_symmetric_mirroring_identity_transform(self, simple_template_params):
        """Test that points equidistant from center mirror to each other."""
        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=2, W=1)
        # Two spatial locations with symmetric ML values
        pred = torch.zeros(1, 3, 2, 1, dtype=torch.float32)
        pred[0, 0, 0, 0] = 20.0  # ML at location (0,0)
        pred[0, 0, 1, 0] = 79.0  # ML at location (1,0)
        pred[0, 1, :, :] = 50.0  # AP channel
        pred[0, 2, :, :] = 50.0  # DV channel

        flipped = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # First location should flip to 79, second to 20
        assert torch.allclose(flipped[0, 0, 0, 0], torch.tensor(79.0), atol=1e-5)
        assert torch.allclose(flipped[0, 0, 1, 0], torch.tensor(20.0), atol=1e-5)

        # Other dimensions unchanged
        assert torch.allclose(flipped[0, 1], pred[0, 1])
        assert torch.allclose(flipped[0, 2], pred[0, 2])

    def test_double_mirror_returns_original(self, simple_template_params):
        """Test that mirroring twice returns to original position."""
        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=5, W=5)
        pred = torch.zeros(1, 3, 5, 5, dtype=torch.float32)
        pred[0, 0, :, :] = 25.0  # ML values
        pred[0, 1, :, :] = 30.0  # AP values
        pred[0, 2, :, :] = 40.0  # DV values

        flipped_once = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        flipped_twice = mirror_points(
            pred=flipped_once,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # Should return to original
        assert torch.allclose(flipped_twice, pred, atol=1e-4)

    def test_only_ml_dimension_affected(self, simple_template_params):
        """Test that only ML (first) dimension is mirrored."""
        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=3, W=3)
        pred = torch.zeros(1, 3, 3, 3, dtype=torch.float32)
        pred[0, 0, :, :] = 25.0  # ML values
        pred[0, 1, :, :] = 30.0  # AP values
        pred[0, 2, :, :] = 40.0  # DV values

        flipped = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # ML dimension should change
        assert not torch.allclose(flipped[0, 0], pred[0, 0])

        # AP and DV dimensions should be unchanged
        assert torch.allclose(flipped[0, 1], pred[0, 1])
        assert torch.allclose(flipped[0, 2], pred[0, 2])

    def test_with_origin_offset(self):
        """Test mirroring with non-zero origin."""
        template_params = AntsImageParameters(
            orientation="RAS",
            dims=3,
            scale=(1.0, 1.0, 1.0),
            origin=(-50.0, -50.0, -50.0),  # Shifted origin
            direction=np.array([1.0, 1.0, 1.0])
        )

        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=2, W=2)
        pred = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
        pred[0, 0, :, :] = 0.0   # ML values in physical space
        pred[0, 1, :, :] = 0.0   # AP values
        pred[0, 2, :, :] = 0.0   # DV values

        flipped = mirror_points(
            pred=pred,
            template_parameters=template_params,
            ml_dim_size=ml_dim_size
        )

        # Should still work correctly with offset
        # In index space: (0 - (-50)) / 1 = 50
        # Flipped: 99 - 50 = 49
        # Back to physical: 49 * 1 + (-50) = -1
        expected_ml = torch.tensor(-1.0)
        assert torch.allclose(flipped[0, 0], expected_ml, atol=1e-4)

    def test_with_scaling(self):
        """Test mirroring with non-unit scaling."""
        template_params = AntsImageParameters(
            orientation="RAS",
            dims=3,
            scale=(0.025, 0.025, 0.025),  # 25 micron spacing
            origin=(0.0, 0.0, 0.0),
            direction=np.array([1.0, 1.0, 1.0])
        )

        ml_dim_size = 528  # Typical CCF dimension

        # Create coordinate field with shape (B=1, C=3, H=2, W=2)
        pred = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
        pred[0, 0, :, :] = 5.0  # ML values in physical space (mm)
        pred[0, 1, :, :] = 4.0  # AP values
        pred[0, 2, :, :] = 6.0  # DV values

        flipped = mirror_points(
            pred=pred,
            template_parameters=template_params,
            ml_dim_size=ml_dim_size
        )

        # Conversion to index: 5.0 / 0.025 = 200
        # Flipped: 527 - 200 = 327
        # Back to physical: 327 * 0.025 = 8.175
        expected_ml = torch.tensor(8.175)
        assert torch.allclose(flipped[0, 0], expected_ml, atol=1e-4)

        # Other dimensions unchanged
        assert torch.allclose(flipped[0, 1], pred[0, 1])
        assert torch.allclose(flipped[0, 2], pred[0, 2])

    def test_with_negative_direction(self):
        """Test mirroring with negative direction."""
        template_params = AntsImageParameters(
            orientation="RAS",
            dims=3,
            scale=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.array([-1.0, 1.0, 1.0])  # Negative ML direction
        )

        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=3, W=3)
        pred = torch.zeros(1, 3, 3, 3, dtype=torch.float32)
        pred[0, 0, :, :] = 25.0  # ML values
        pred[0, 1, :, :] = 30.0  # AP values
        pred[0, 2, :, :] = 40.0  # DV values

        flipped = mirror_points(
            pred=pred,
            template_parameters=template_params,
            ml_dim_size=ml_dim_size
        )

        # Should handle negative direction correctly
        assert isinstance(flipped, torch.Tensor)
        assert flipped.shape == pred.shape

    def test_batch_processing(self, simple_template_params):
        """Test that multiple batches are processed correctly."""
        ml_dim_size = 100

        # Create coordinate field with shape (B=3, C=3, H=2, W=2)
        # Multiple batches with different ML values
        pred = torch.zeros(3, 3, 2, 2, dtype=torch.float32)
        pred[0, 0, :, :] = 10.0  # Batch 0 ML values
        pred[1, 0, :, :] = 50.0  # Batch 1 ML values
        pred[2, 0, :, :] = 90.0  # Batch 2 ML values
        pred[:, 1, :, :] = 40.0  # AP values for all batches
        pred[:, 2, :, :] = 60.0  # DV values for all batches

        flipped = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # Should process all batches
        assert flipped.shape == pred.shape

        # Each batch should be mirrored independently
        for i in range(pred.shape[0]):
            single_flipped = mirror_points(
                pred=pred[i:i+1],
                template_parameters=simple_template_params,
                ml_dim_size=ml_dim_size
            )
            assert torch.allclose(flipped[i], single_flipped[0])

    def test_edge_points(self, simple_template_params):
        """Test mirroring of edge points (0 and max)."""
        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=2, W=1)
        # Two spatial locations with edge ML values
        pred = torch.zeros(1, 3, 2, 1, dtype=torch.float32)
        pred[0, 0, 0, 0] = 0.0    # Left edge ML
        pred[0, 0, 1, 0] = 99.0   # Right edge ML
        pred[0, 1, :, :] = 50.0   # AP values
        pred[0, 2, :, :] = 50.0   # DV values

        flipped = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # Left edge should flip to right edge
        assert torch.allclose(flipped[0, 0, 0, 0], torch.tensor(99.0), atol=1e-5)
        # Right edge should flip to left edge
        assert torch.allclose(flipped[0, 0, 1, 0], torch.tensor(0.0), atol=1e-5)

    def test_original_tensor_unchanged(self, simple_template_params):
        """Test that original tensor is not modified (uses clone)."""
        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=3, W=3)
        pred = torch.zeros(1, 3, 3, 3, dtype=torch.float32)
        pred[0, 0, :, :] = 25.0  # ML values
        pred[0, 1, :, :] = 30.0  # AP values
        pred[0, 2, :, :] = 40.0  # DV values
        pred_original = pred.clone()

        flipped = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # Original should be unchanged
        assert torch.allclose(pred, pred_original)
        # Flipped should be different
        assert not torch.allclose(flipped, pred)

    def test_realistic_ccf_scenario(self, complex_template_params):
        """Test with realistic CCF Allen Brain Atlas parameters."""
        ml_dim_size = 528  # CCF ML dimension size

        # Create coordinate field with shape (B=1, C=3, H=10, W=10)
        # Typical points in CCF physical space (in mm)
        pred = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
        pred[0, 0, :5, :] = -3.5  # ML values for first half
        pred[0, 0, 5:, :] = 2.1   # ML values for second half
        pred[0, 1, :5, :] = 4.2   # AP values for first half
        pred[0, 1, 5:, :] = -2.3  # AP values for second half
        pred[0, 2, :5, :] = -1.5  # DV values for first half
        pred[0, 2, 5:, :] = 3.8   # DV values for second half

        flipped = mirror_points(
            pred=pred,
            template_parameters=complex_template_params,
            ml_dim_size=ml_dim_size
        )

        # Verify shape
        assert flipped.shape == pred.shape

        # Verify AP and DV unchanged
        assert torch.allclose(flipped[0, 1], pred[0, 1], atol=1e-5)
        assert torch.allclose(flipped[0, 2], pred[0, 2], atol=1e-5)

        # Verify ML is different (mirrored)
        assert not torch.allclose(flipped[0, 0], pred[0, 0])

    def test_dtype_preservation(self, simple_template_params):
        """Test that tensor dtype is preserved."""
        ml_dim_size = 100

        # Test with float32
        pred_f32 = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
        pred_f32[0, 0, :, :] = 25.0
        pred_f32[0, 1, :, :] = 30.0
        pred_f32[0, 2, :, :] = 40.0
        flipped_f32 = mirror_points(
            pred=pred_f32,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )
        assert flipped_f32.dtype == torch.float32

        # Test with float64
        pred_f64 = torch.zeros(1, 3, 2, 2, dtype=torch.float64)
        pred_f64[0, 0, :, :] = 25.0
        pred_f64[0, 1, :, :] = 30.0
        pred_f64[0, 2, :, :] = 40.0
        flipped_f64 = mirror_points(
            pred=pred_f64,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )
        assert flipped_f64.dtype == torch.float64

    def test_device_preservation(self, simple_template_params):
        """Test that tensor device is preserved."""
        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=2, W=2)
        pred = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
        pred[0, 0, :, :] = 25.0
        pred[0, 1, :, :] = 30.0
        pred[0, 2, :, :] = 40.0

        flipped = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # Should be on same device
        assert flipped.device == pred.device

    def test_gradient_flow(self, simple_template_params):
        """Test that gradients can flow through the operation."""
        ml_dim_size = 100

        # Create coordinate field with shape (B=1, C=3, H=2, W=2)
        pred = torch.zeros(1, 3, 2, 2, dtype=torch.float32, requires_grad=True)
        pred.data[0, 0, :, :] = 25.0
        pred.data[0, 1, :, :] = 30.0
        pred.data[0, 2, :, :] = 40.0

        flipped = mirror_points(
            pred=pred,
            template_parameters=simple_template_params,
            ml_dim_size=ml_dim_size
        )

        # Compute a loss
        loss = flipped.sum()
        loss.backward()

        # Gradient should exist
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
