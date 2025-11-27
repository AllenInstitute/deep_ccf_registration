import numpy as np
import torch
import pytest
from unittest.mock import Mock, patch

from deep_ccf_registration.inference import (
    _resize_to_original,
    _build_class_mapping,
    _update_confusion_matrix,
    _calc_dice_from_confusion_matrix,
    RegionAcronymCCFIdsMap,
)


class TestResizeToOriginal:
    """Tests for the _resize_to_original function."""

    def test_no_padding_no_resize(self):
        """Test when image is already correct size with no padding."""
        img = torch.rand(3, 10, 20).numpy()
        gt_shape = (10, 20)

        result = _resize_to_original(
            img=img,
            gt_shape=gt_shape,
            pre_pad_shape=None,
            pad_top=None,
            pad_left=None
        )

        # Should return same shape, just permuted back
        assert result.shape == (3, 10, 20)
        assert isinstance(result, torch.Tensor)

    def test_with_padding_crop(self):
        """Test cropping out padding."""
        # Create padded image (3, 15, 25) with actual content (10, 20)
        img = torch.zeros(3, 15, 25).numpy()
        # Put data in the center
        img[:, 2:12, 3:23] = 1.0

        result = _resize_to_original(
            img=img,
            gt_shape=(10, 20),
            pre_pad_shape=(10, 20),
            pad_top=2,
            pad_left=3
        )

        # Should extract the (10, 20) region
        assert result.shape == (3, 10, 20)
        # All values should be 1.0 (the data region)
        assert torch.all(result == 1.0)

    def test_with_resize(self):
        """Test resizing to different dimensions."""
        # Start with smaller image
        img = torch.ones(3, 5, 10).numpy()
        gt_shape = (10, 20)

        result = _resize_to_original(
            img=img,
            gt_shape=gt_shape,
            pre_pad_shape=None,
            pad_top=None,
            pad_left=None
        )

        # Should be resized to gt_shape
        assert result.shape == (3, 10, 20)

    def test_permutation_order(self):
        """Test that permutation is correctly applied."""
        # Create image with distinct values per channel
        img = np.zeros((3, 4, 5))
        img[0, :, :] = 1.0
        img[1, :, :] = 2.0
        img[2, :, :] = 3.0

        result = _resize_to_original(
            img=img,
            gt_shape=(4, 5),
            pre_pad_shape=None,
            pad_top=None,
            pad_left=None
        )

        # Check channels are preserved
        assert torch.all(result[0] == 1.0)
        assert torch.all(result[1] == 2.0)
        assert torch.all(result[2] == 3.0)

    def test_padding_and_resize_combined(self):
        """Test both padding removal and resizing."""
        # Padded image
        img = torch.zeros(3, 12, 22).numpy()
        img[:, 1:11, 1:21] = 1.0

        result = _resize_to_original(
            img=img,
            gt_shape=(20, 40),  # Resize to larger
            pre_pad_shape=(10, 20),
            pad_top=1,
            pad_left=1
        )

        # Should be resized to gt_shape
        assert result.shape == (3, 20, 40)


class TestBuildClassMapping:
    """Tests for the _build_class_mapping function."""

    def test_basic_mapping(self):
        """Test basic class mapping creation."""
        region_map = {
            'CTX': [1, 2, 3],
            'HIP': [4, 5],
            'TH': [6]
        }

        mapping = _build_class_mapping(region_map)

        # CTX should be class 1
        assert mapping[1] == 1
        assert mapping[2] == 1
        assert mapping[3] == 1

        # HIP should be class 2
        assert mapping[4] == 2
        assert mapping[5] == 2

        # TH should be class 3
        assert mapping[6] == 3

    def test_empty_region_map(self):
        """Test with empty region map."""
        region_map = {}
        mapping = _build_class_mapping(region_map)

        assert mapping == {}

    def test_single_region(self):
        """Test with single region."""
        region_map = {'CTX': [1, 2, 3, 4, 5]}
        mapping = _build_class_mapping(region_map)

        # All should map to class 1
        for i in range(1, 6):
            assert mapping[i] == 1

    def test_class_indices_start_at_one(self):
        """Test that class indices start at 1 (0 reserved for background)."""
        region_map = {
            'A': [1],
            'B': [2],
            'C': [3]
        }

        mapping = _build_class_mapping(region_map)

        # First region should be class 1, not 0
        assert mapping[1] == 1
        assert mapping[2] == 2
        assert mapping[3] == 3


class TestUpdateConfusionMatrix:
    """Tests for the _update_confusion_matrix function."""

    def test_perfect_prediction(self):
        """Test with perfect predictions."""
        confusion_matrix = np.zeros((3, 3), dtype=np.int64)
        pred_annotations = np.array([[1, 1], [2, 2]])
        true_annotations = np.array([[1, 1], [2, 2]])
        class_mapping = {1: 1, 2: 2}

        _update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            pred_annotations=pred_annotations,
            true_annotations=true_annotations,
            class_mapping=class_mapping
        )

        # Diagonal should have counts
        assert confusion_matrix[1, 1] == 2  # Class 1 correct
        assert confusion_matrix[2, 2] == 2  # Class 2 correct
        # Off-diagonal should be zero
        assert confusion_matrix[1, 2] == 0
        assert confusion_matrix[2, 1] == 0

    def test_with_errors(self):
        """Test with some misclassifications."""
        confusion_matrix = np.zeros((3, 3), dtype=np.int64)
        pred_annotations = np.array([[1, 2], [1, 2]])
        true_annotations = np.array([[1, 1], [2, 2]])
        class_mapping = {1: 1, 2: 2}

        _update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            pred_annotations=pred_annotations,
            true_annotations=true_annotations,
            class_mapping=class_mapping
        )

        # True class 1, predicted class 1: 1 sample
        assert confusion_matrix[1, 1] == 1
        # True class 1, predicted class 2: 1 sample
        assert confusion_matrix[1, 2] == 1
        # True class 2, predicted class 1: 1 sample
        assert confusion_matrix[2, 1] == 1
        # True class 2, predicted class 2: 1 sample
        assert confusion_matrix[2, 2] == 1

    def test_background_class(self):
        """Test that unmapped IDs default to background (class 0)."""
        confusion_matrix = np.zeros((3, 3), dtype=np.int64)
        pred_annotations = np.array([[0, 1], [999, 1]])
        true_annotations = np.array([[0, 1], [999, 1]])
        class_mapping = {1: 1}  # Only class 1 is mapped

        _update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            pred_annotations=pred_annotations,
            true_annotations=true_annotations,
            class_mapping=class_mapping
        )

        # Background (ID 0 and 999 -> class 0)
        assert confusion_matrix[0, 0] == 2  # Two background samples
        # Class 1
        assert confusion_matrix[1, 1] == 2  # Two class 1 samples

    def test_accumulation_across_calls(self):
        """Test that matrix accumulates across multiple calls."""
        confusion_matrix = np.zeros((3, 3), dtype=np.int64)
        class_mapping = {1: 1, 2: 2}

        # First call
        _update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            pred_annotations=np.array([[1]]),
            true_annotations=np.array([[1]]),
            class_mapping=class_mapping
        )

        # Second call
        _update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            pred_annotations=np.array([[1]]),
            true_annotations=np.array([[1]]),
            class_mapping=class_mapping
        )

        # Should accumulate
        assert confusion_matrix[1, 1] == 2


class TestCalcDiceFromConfusionMatrix:
    """Tests for the _calc_dice_from_confusion_matrix function."""

    def test_perfect_dice(self):
        """Test Dice score with perfect predictions."""
        # Perfect confusion matrix
        confusion_matrix = np.array([
            [0, 0, 0],  # Background
            [0, 10, 0],  # Class 1: 10 correct
            [0, 0, 20],  # Class 2: 20 correct
        ])

        region_map = {
            'CTX': [1, 2, 3],
            'HIP': [4, 5]
        }

        dice_scores = _calc_dice_from_confusion_matrix(
            confusion_matrix=confusion_matrix,
            region_map=region_map
        )

        # Perfect predictions should have Dice = 1.0
        assert dice_scores['CTX'] == 1.0
        assert dice_scores['HIP'] == 1.0

    def test_zero_dice(self):
        """Test Dice score with completely wrong predictions."""
        confusion_matrix = np.array([
            [0, 0, 0],
            [0, 0, 10],  # True class 1, predicted as class 2
            [0, 10, 0],  # True class 2, predicted as class 1
        ])

        region_map = {
            'CTX': [1],
            'HIP': [2]
        }

        dice_scores = _calc_dice_from_confusion_matrix(
            confusion_matrix=confusion_matrix,
            region_map=region_map
        )

        # No correct predictions, Dice = 0
        assert dice_scores['CTX'] == 0.0
        assert dice_scores['HIP'] == 0.0

    def test_partial_dice(self):
        """Test Dice score with partial correctness."""
        # Class 1: TP=5, FP=3, FN=2
        # Class 2: TP=10, FP=2, FN=3
        confusion_matrix = np.array([
            [0, 0, 0],
            [0, 5, 2],   # Row for true class 1: TP=5, FN=2
            [0, 3, 10],  # Row for true class 2: TP=10, FN=3
        ])
        # Column sums: class 1 predicted = 5+3=8, so FP for class 1 = 8-5=3
        # Column sums: class 2 predicted = 2+10=12, so FP for class 2 = 12-10=2

        region_map = {
            'CTX': [1],
            'HIP': [2]
        }

        dice_scores = _calc_dice_from_confusion_matrix(
            confusion_matrix=confusion_matrix,
            region_map=region_map
        )

        # CTX: Dice = 2*TP / (2*TP + FP + FN) = 2*5 / (10 + 3 + 2) = 10/15 = 0.666...
        assert abs(dice_scores['CTX'] - 2 / 3) < 0.001

        # HIP: Dice = 2*10 / (20 + 2 + 3) = 20/25 = 0.8
        assert abs(dice_scores['HIP'] - 0.8) < 0.001

    def test_empty_class_skipped(self):
        """Test that classes with no samples are skipped."""
        confusion_matrix = np.array([
            [0, 0, 0],
            [0, 10, 0],
            [0, 0, 0],  # Empty class
        ])

        region_map = {
            'CTX': [1],
            'HIP': [2]  # This class has no samples
        }

        dice_scores = _calc_dice_from_confusion_matrix(
            confusion_matrix=confusion_matrix,
            region_map=region_map
        )

        # Only CTX should be in results
        assert 'CTX' in dice_scores
        assert 'HIP' not in dice_scores  # Skipped because empty

    def test_dice_formula(self):
        """Test the Dice coefficient formula: 2*TP / (2*TP + FP + FN)."""
        # Simple case: TP=6, FP=2, FN=4
        confusion_matrix = np.array([
            [0, 0],
            [0, 6],  # True class 1, predicted class 1: 6
        ])
        # Add FN (true class 1, predicted other): 4
        confusion_matrix[1, 0] = 4  # This creates FN=4
        # We need total predicted as class 1 to be 6+2=8
        confusion_matrix[0, 1] = 2  # This creates FP=2

        region_map = {'CTX': [1]}

        dice_scores = _calc_dice_from_confusion_matrix(
            confusion_matrix=confusion_matrix,
            region_map=region_map
        )

        # Dice = 2*6 / (2*6 + 2 + 4) = 12 / 18 = 2/3
        expected_dice = 2 * 6 / (2 * 6 + 2 + 4)
        assert abs(dice_scores['CTX'] - expected_dice) < 0.001


class TestRegionAcronymCCFIdsMap:
    """Tests for the RegionAcronymCCFIdsMap pydantic model."""

    def test_creation(self):
        """Test basic creation of RegionAcronymCCFIdsMap."""
        region_map = RegionAcronymCCFIdsMap(
            major_regions={'CTX': [1, 2, 3], 'HIP': [4, 5]},
            small_regions={'CTXa': [1], 'CTXb': [2, 3], 'HIPa': [4, 5]}
        )

        assert region_map.major_regions == {'CTX': [1, 2, 3], 'HIP': [4, 5]}
        assert region_map.small_regions == {'CTXa': [1], 'CTXb': [2, 3], 'HIPa': [4, 5]}

    def test_empty_regions(self):
        """Test with empty region dictionaries."""
        region_map = RegionAcronymCCFIdsMap(
            major_regions={},
            small_regions={}
        )

        assert region_map.major_regions == {}
        assert region_map.small_regions == {}

    def test_validation(self):
        """Test pydantic validation."""
        # Valid
        region_map = RegionAcronymCCFIdsMap(
            major_regions={'CTX': [1, 2, 3]},
            small_regions={'CTXa': [1]}
        )
        assert isinstance(region_map, RegionAcronymCCFIdsMap)

    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        region_map = RegionAcronymCCFIdsMap(
            major_regions={'CTX': [1, 2, 3]},
            small_regions={'CTXa': [1]}
        )

        # Should be serializable
        json_str = region_map.model_dump_json()
        assert isinstance(json_str, str)

        # Should be deserializable
        region_map2 = RegionAcronymCCFIdsMap.model_validate_json(json_str)
        assert region_map2.major_regions == region_map.major_regions
        assert region_map2.small_regions == region_map.small_regions
