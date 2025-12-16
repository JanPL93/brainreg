"""
Landmark-based correction for atlas registration using thin-plate splines.

This module provides functionality to correct registration errors by allowing
users to define correspondence points between the registered atlas and the
actual brain data. The correction can be applied either:
1. Post-registration: Apply a thin-plate spline warp to the registered atlas
2. Pre-registration: Warp the input data to better match the atlas before
   running the registration pipeline
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ThinPlateSplineTransform:
    """
    Thin-plate spline (TPS) transformation for landmark-based image warping.

    TPS provides a smooth interpolation between control points, making it
    ideal for correcting local registration errors while maintaining
    global smoothness.

    Parameters
    ----------
    source_points : np.ndarray
        Source landmark coordinates, shape (N, 3) for 3D or (N, 2) for 2D.
    target_points : np.ndarray
        Target landmark coordinates, same shape as source_points.
    smoothing : float, optional
        Smoothing parameter for the RBF interpolation. Higher values produce
        smoother transformations but may not pass exactly through control
        points. Default is 0.0 (exact interpolation).
    """

    def __init__(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        smoothing: float = 0.0,
    ):
        self.source_points = np.asarray(source_points, dtype=np.float64)
        self.target_points = np.asarray(target_points, dtype=np.float64)
        self.smoothing = smoothing
        self.ndim = self.source_points.shape[1]

        if self.source_points.shape != self.target_points.shape:
            raise ValueError(
                "Source and target points must have the same shape"
            )

        if len(self.source_points) < 4:
            raise ValueError(
                "At least 4 correspondence points are required for "
                "thin-plate spline interpolation"
            )

        # Build RBF interpolators for each dimension
        # Maps from target space back to source space for image resampling
        self._interpolators = []
        for dim in range(self.ndim):
            interp = RBFInterpolator(
                self.target_points,
                self.source_points[:, dim],
                kernel="thin_plate_spline",
                smoothing=self.smoothing,
            )
            self._interpolators.append(interp)

    def __call__(
        self, points: np.ndarray
    ) -> np.ndarray:
        """
        Transform points from target space to source space.

        Parameters
        ----------
        points : np.ndarray
            Points to transform, shape (M, ndim) or (ndim,).

        Returns
        -------
        np.ndarray
            Transformed points with same shape as input.
        """
        points = np.atleast_2d(points)
        result = np.column_stack(
            [interp(points) for interp in self._interpolators]
        )
        return result.squeeze()

    def compute_displacement_field(
        self, shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Compute the full displacement field for an image of given shape.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output displacement field (excluding the last
            dimension which will be ndim).

        Returns
        -------
        np.ndarray
            Displacement field of shape (*shape, ndim), where each voxel
            contains the displacement vector.
        """
        # Create coordinate grid
        coords = np.meshgrid(
            *[np.arange(s) for s in shape],
            indexing='ij'
        )
        grid_points = np.column_stack([c.ravel() for c in coords])

        # Transform all points
        transformed = self(grid_points)

        # Compute displacement
        displacement = transformed - grid_points

        # Reshape to image shape
        displacement_field = displacement.reshape(*shape, self.ndim)
        return displacement_field

    def to_dict(self) -> dict:
        """Serialize the transform to a dictionary."""
        return {
            "source_points": self.source_points.tolist(),
            "target_points": self.target_points.tolist(),
            "smoothing": self.smoothing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ThinPlateSplineTransform":
        """Create a transform from a dictionary."""
        return cls(
            source_points=np.array(data["source_points"]),
            target_points=np.array(data["target_points"]),
            smoothing=data.get("smoothing", 0.0),
        )


def apply_landmark_correction(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    smoothing: float = 0.0,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    chunk_size: int = 50,
) -> np.ndarray:
    """
    Apply landmark-based correction to an image using thin-plate splines.

    This function warps the input image so that the source landmarks move
    to the target landmark positions.

    Parameters
    ----------
    image : np.ndarray
        Input image to warp (2D or 3D).
    source_points : np.ndarray
        Current landmark positions in the image, shape (N, ndim).
    target_points : np.ndarray
        Desired landmark positions, shape (N, ndim).
    smoothing : float, optional
        TPS smoothing parameter. Default is 0.0.
    order : int, optional
        Interpolation order for image resampling (0-5). Default is 1 (linear).
    mode : str, optional
        How to handle boundaries. Default is "constant".
    cval : float, optional
        Value to use for out-of-bounds points. Default is 0.0.
    chunk_size : int, optional
        Process the image in chunks to reduce memory usage. Default is 50.

    Returns
    -------
    np.ndarray
        The warped image with the same shape and dtype as input.
    """
    original_dtype = image.dtype
    ndim = image.ndim

    # Create the TPS transform
    tps = ThinPlateSplineTransform(source_points, target_points, smoothing)

    # Process in chunks to manage memory for large 3D images
    output = np.zeros_like(image, dtype=np.float64)

    if ndim == 3:
        # Process slice by slice for memory efficiency
        for z_start in tqdm(
            range(0, image.shape[0], chunk_size),
            desc="Applying correction",
            unit="chunk"
        ):
            z_end = min(z_start + chunk_size, image.shape[0])

            # Create coordinate grid for this chunk
            zz, yy, xx = np.meshgrid(
                np.arange(z_start, z_end),
                np.arange(image.shape[1]),
                np.arange(image.shape[2]),
                indexing='ij'
            )

            chunk_coords = np.column_stack([
                zz.ravel(), yy.ravel(), xx.ravel()
            ])

            # Transform coordinates
            src_coords = tps(chunk_coords)

            # Resample image at transformed coordinates
            src_coords_t = src_coords.T.reshape(3, z_end - z_start, *image.shape[1:])

            chunk_output = map_coordinates(
                image.astype(np.float64),
                src_coords_t,
                order=order,
                mode=mode,
                cval=cval,
            )

            output[z_start:z_end] = chunk_output
    else:
        # 2D case
        yy, xx = np.meshgrid(
            np.arange(image.shape[0]),
            np.arange(image.shape[1]),
            indexing='ij'
        )
        coords = np.column_stack([yy.ravel(), xx.ravel()])
        src_coords = tps(coords)
        src_coords_t = src_coords.T.reshape(2, *image.shape)

        output = map_coordinates(
            image.astype(np.float64),
            src_coords_t,
            order=order,
            mode=mode,
            cval=cval,
        )

    # Convert back to original dtype
    if np.issubdtype(original_dtype, np.integer):
        output = np.clip(output, np.iinfo(original_dtype).min,
                         np.iinfo(original_dtype).max)
    return output.astype(original_dtype)


def correct_registered_atlas(
    registered_atlas: np.ndarray,
    atlas_landmarks: np.ndarray,
    data_landmarks: np.ndarray,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Correct a registered atlas using user-defined correspondence points.

    Use this when the initial registration has errors that need to be
    corrected. The user marks corresponding points on the atlas and the
    brain data, and this function warps the atlas to better match.

    Parameters
    ----------
    registered_atlas : np.ndarray
        The registered atlas labels (from brainreg output).
    atlas_landmarks : np.ndarray
        Landmark points on the registered atlas, shape (N, 3).
    data_landmarks : np.ndarray
        Corresponding landmark points on the brain data, shape (N, 3).
    smoothing : float, optional
        TPS smoothing parameter. Higher values produce smoother corrections.
        Default is 0.0 for exact landmark matching.

    Returns
    -------
    np.ndarray
        The corrected atlas with landmarks aligned to data landmarks.
    """
    # For label images, use nearest-neighbor interpolation
    return apply_landmark_correction(
        registered_atlas,
        source_points=atlas_landmarks,
        target_points=data_landmarks,
        smoothing=smoothing,
        order=0,  # Nearest neighbor for labels
        mode="constant",
        cval=0,  # Background label
    )


def warp_data_for_registration(
    brain_data: np.ndarray,
    data_landmarks: np.ndarray,
    atlas_landmarks: np.ndarray,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Pre-warp brain data to better match atlas before registration.

    Use this when you want to help the registration pipeline by
    pre-aligning the data. This can be useful when automatic registration
    struggles due to low contrast or unusual brain geometry.

    Parameters
    ----------
    brain_data : np.ndarray
        The brain image data to warp.
    data_landmarks : np.ndarray
        Landmark points on the brain data, shape (N, 3).
    atlas_landmarks : np.ndarray
        Corresponding landmark points on the atlas (or where you want
        the data landmarks to move to), shape (N, 3).
    smoothing : float, optional
        TPS smoothing parameter. Default is 0.0.

    Returns
    -------
    np.ndarray
        The pre-warped brain data ready for registration.
    """
    return apply_landmark_correction(
        brain_data,
        source_points=data_landmarks,
        target_points=atlas_landmarks,
        smoothing=smoothing,
        order=1,  # Linear interpolation for intensity images
        mode="constant",
        cval=0,
    )


def save_landmarks(
    filepath: Union[str, Path],
    atlas_landmarks: np.ndarray,
    data_landmarks: np.ndarray,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save landmark correspondences to a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    atlas_landmarks : np.ndarray
        Landmark points on the atlas, shape (N, 3).
    data_landmarks : np.ndarray
        Corresponding landmark points on the data, shape (N, 3).
    metadata : dict, optional
        Additional metadata to save (e.g., registration parameters used).
    """
    filepath = Path(filepath)
    data = {
        "atlas_landmarks": atlas_landmarks.tolist(),
        "data_landmarks": data_landmarks.tolist(),
        "n_landmarks": len(atlas_landmarks),
    }
    if metadata:
        data["metadata"] = metadata

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(atlas_landmarks)} landmarks to {filepath}")


def load_landmarks(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """
    Load landmark correspondences from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    atlas_landmarks : np.ndarray
        Landmark points on the atlas, shape (N, 3).
    data_landmarks : np.ndarray
        Corresponding landmark points on the data, shape (N, 3).
    metadata : dict or None
        Additional metadata if present.
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        data = json.load(f)

    atlas_landmarks = np.array(data["atlas_landmarks"])
    data_landmarks = np.array(data["data_landmarks"])
    metadata = data.get("metadata")

    logger.info(f"Loaded {len(atlas_landmarks)} landmarks from {filepath}")
    return atlas_landmarks, data_landmarks, metadata


def compute_registration_error(
    atlas_landmarks: np.ndarray,
    data_landmarks: np.ndarray,
) -> dict:
    """
    Compute statistics about registration error from landmark pairs.

    Parameters
    ----------
    atlas_landmarks : np.ndarray
        Landmark points on the atlas, shape (N, 3).
    data_landmarks : np.ndarray
        Corresponding landmark points on the data, shape (N, 3).

    Returns
    -------
    dict
        Dictionary containing error statistics:
        - distances: array of Euclidean distances for each landmark pair
        - mean_error: mean distance
        - max_error: maximum distance
        - std_error: standard deviation of distances
        - median_error: median distance
    """
    distances = np.linalg.norm(atlas_landmarks - data_landmarks, axis=1)
    return {
        "distances": distances,
        "mean_error": float(np.mean(distances)),
        "max_error": float(np.max(distances)),
        "std_error": float(np.std(distances)),
        "median_error": float(np.median(distances)),
        "n_landmarks": len(distances),
    }
