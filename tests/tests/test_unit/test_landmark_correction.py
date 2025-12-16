"""
Tests for landmark-based registration correction.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from brainreg.core.utils.landmark_correction import (
    ThinPlateSplineTransform,
    apply_landmark_correction,
    compute_registration_error,
    correct_registered_atlas,
    load_landmarks,
    save_landmarks,
    warp_data_for_registration,
)


class TestThinPlateSplineTransform:
    """Tests for ThinPlateSplineTransform class."""

    def test_init_basic(self):
        """Test basic initialization with valid points."""
        source = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [10, 10, 10],
        ])
        target = source + 2  # Simple translation

        tps = ThinPlateSplineTransform(source, target)
        assert tps.ndim == 3
        assert len(tps.source_points) == 5

    def test_init_requires_min_points(self):
        """Test that at least 4 points are required."""
        source = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        target = source.copy()

        with pytest.raises(ValueError, match="At least 4"):
            ThinPlateSplineTransform(source, target)

    def test_init_shape_mismatch(self):
        """Test that source and target must have same shape."""
        source = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        target = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        with pytest.raises(ValueError, match="same shape"):
            ThinPlateSplineTransform(source, target)

    def test_identity_transform(self):
        """Test that identical source/target gives identity."""
        points = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [5, 5, 5],
        ])

        tps = ThinPlateSplineTransform(points, points.copy())

        # Test point at known location
        test_point = np.array([5, 5, 5])
        result = tps(test_point)
        np.testing.assert_array_almost_equal(result, test_point, decimal=5)

    def test_translation_transform(self):
        """Test a simple translation is captured correctly."""
        source = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [10, 10, 10],
        ])
        offset = np.array([5, 3, 2])
        target = source + offset

        tps = ThinPlateSplineTransform(source, target)

        # Test that a point in the middle transforms correctly
        test_point = np.array([5, 5, 5])
        result = tps(test_point)

        # Result should be approximately test_point - offset
        # (we map from target to source for resampling)
        expected = test_point - offset
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        source = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
        ])
        target = source + 1

        tps = ThinPlateSplineTransform(source, target, smoothing=0.5)
        data = tps.to_dict()

        tps2 = ThinPlateSplineTransform.from_dict(data)

        np.testing.assert_array_equal(tps2.source_points, source)
        np.testing.assert_array_equal(tps2.target_points, target)
        assert tps2.smoothing == 0.5


class TestApplyLandmarkCorrection:
    """Tests for apply_landmark_correction function."""

    def test_identity_correction_3d(self):
        """Test that identity landmarks don't change the image."""
        image = np.random.rand(20, 20, 20).astype(np.float32)

        points = np.array([
            [2, 2, 2],
            [2, 2, 17],
            [2, 17, 2],
            [17, 2, 2],
            [10, 10, 10],
        ])

        result = apply_landmark_correction(
            image,
            source_points=points,
            target_points=points.copy(),
        )

        # Should be nearly identical
        assert result.shape == image.shape
        # Central region should be preserved
        np.testing.assert_array_almost_equal(
            result[5:15, 5:15, 5:15],
            image[5:15, 5:15, 5:15],
            decimal=4
        )

    def test_preserves_dtype(self):
        """Test that output dtype matches input."""
        image = np.random.randint(0, 255, (20, 20, 20), dtype=np.uint8)

        points = np.array([
            [2, 2, 2],
            [17, 2, 2],
            [2, 17, 2],
            [2, 2, 17],
        ])

        result = apply_landmark_correction(
            image,
            source_points=points,
            target_points=points + 1,
        )

        assert result.dtype == np.uint8


class TestCorrectRegisteredAtlas:
    """Tests for correct_registered_atlas function."""

    def test_correction_uses_nearest_neighbor(self):
        """Test that atlas labels use nearest-neighbor interpolation."""
        # Create a simple labeled atlas
        atlas = np.zeros((20, 20, 20), dtype=np.int32)
        atlas[:10, :, :] = 1
        atlas[10:, :, :] = 2

        # Define landmarks
        atlas_landmarks = np.array([
            [5, 10, 10],
            [15, 10, 10],
            [10, 5, 10],
            [10, 15, 10],
        ])
        data_landmarks = atlas_landmarks.copy()

        result = correct_registered_atlas(
            atlas, atlas_landmarks, data_landmarks
        )

        # Result should only contain original label values
        unique_values = np.unique(result)
        assert all(v in [0, 1, 2] for v in unique_values)


class TestWarpDataForRegistration:
    """Tests for warp_data_for_registration function."""

    def test_warp_preserves_range(self):
        """Test that warping preserves intensity range."""
        data = np.random.rand(20, 20, 20) * 1000
        data = data.astype(np.float32)

        landmarks = np.array([
            [5, 5, 5],
            [5, 5, 15],
            [5, 15, 5],
            [15, 5, 5],
        ])

        result = warp_data_for_registration(
            data,
            data_landmarks=landmarks,
            atlas_landmarks=landmarks + 1,
        )

        # Values should be reasonable
        assert result.min() >= 0
        assert result.max() <= data.max() * 1.5  # Allow some interpolation overshoot


class TestSaveLoadLandmarks:
    """Tests for landmark I/O functions."""

    def test_save_load_roundtrip(self):
        """Test that landmarks can be saved and loaded."""
        atlas_landmarks = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        data_landmarks = np.array([
            [1.5, 2.5, 3.5],
            [4.5, 5.5, 6.5],
            [7.5, 8.5, 9.5],
        ])
        metadata = {"atlas": "allen_mouse_25um"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "landmarks.json"

            save_landmarks(filepath, atlas_landmarks, data_landmarks, metadata)

            # Verify file exists and is valid JSON
            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert data["n_landmarks"] == 3

            # Load and verify
            loaded_atlas, loaded_data, loaded_meta = load_landmarks(filepath)

            np.testing.assert_array_almost_equal(loaded_atlas, atlas_landmarks)
            np.testing.assert_array_almost_equal(loaded_data, data_landmarks)
            assert loaded_meta["atlas"] == "allen_mouse_25um"


class TestComputeRegistrationError:
    """Tests for compute_registration_error function."""

    def test_perfect_alignment(self):
        """Test that identical points give zero error."""
        points = np.array([
            [0, 0, 0],
            [10, 10, 10],
            [5, 5, 5],
        ])

        stats = compute_registration_error(points, points.copy())

        assert stats["mean_error"] == 0.0
        assert stats["max_error"] == 0.0
        assert stats["n_landmarks"] == 3

    def test_known_displacement(self):
        """Test with known displacement."""
        atlas = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
        ])
        # Displace by [3, 4, 0] -> distance = 5
        data = atlas + np.array([3, 4, 0])

        stats = compute_registration_error(atlas, data)

        assert stats["mean_error"] == 5.0
        assert stats["max_error"] == 5.0
        assert stats["median_error"] == 5.0


class TestThinPlateSpline2D:
    """Tests for 2D thin-plate spline functionality."""

    def test_2d_transform(self):
        """Test TPS with 2D points."""
        source = np.array([
            [0, 0],
            [10, 0],
            [0, 10],
            [10, 10],
            [5, 5],
        ])
        target = source + 2

        tps = ThinPlateSplineTransform(source, target)
        assert tps.ndim == 2

        result = tps(np.array([5, 5]))
        np.testing.assert_array_almost_equal(result, [3, 3], decimal=3)

    def test_2d_image_correction(self):
        """Test applying correction to 2D image."""
        image = np.random.rand(50, 50).astype(np.float32)

        points = np.array([
            [5, 5],
            [5, 45],
            [45, 5],
            [45, 45],
        ])

        result = apply_landmark_correction(
            image,
            source_points=points,
            target_points=points,
        )

        assert result.shape == image.shape
        assert result.dtype == image.dtype
