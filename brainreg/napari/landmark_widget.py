"""
Napari widget for landmark-based registration correction.

This widget allows users to:
1. Load a completed brainreg registration
2. Add correspondence points where registration errors are visible
3. Apply thin-plate spline corrections to improve alignment
4. Optionally pre-warp data for re-registration
"""

import json
import logging
import pathlib
from enum import Enum
from typing import Optional

import napari
import numpy as np
import tifffile
from brainglobe_utils.qtpy.logo import header_widget
from magicgui import magicgui
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from brainreg.core.utils.landmark_correction import (
    compute_registration_error,
    correct_registered_atlas,
    load_landmarks,
    save_landmarks,
    warp_data_for_registration,
)
from brainreg.napari.register import add_registered_image_layers

logger = logging.getLogger(__name__)


class CorrectionMode(Enum):
    """Correction mode options."""

    POST_REGISTRATION = "Correct atlas (post-registration)"
    PRE_REGISTRATION = "Warp data (pre-registration)"


def brainreg_landmark_correction():
    """
    Create the landmark correction widget.

    Returns
    -------
    magicgui widget
        The configured widget for landmark-based correction.
    """

    @magicgui(
        call_button=False,
        registration_folder=dict(
            mode="d",
            label="Registration folder",
            tooltip="Folder containing brainreg output (with brainreg.json)",
        ),
        load_registration_button=dict(
            widget_type="PushButton",
            text="Load Registration",
        ),
        atlas_layer=dict(
            label="Atlas layer",
            tooltip="The registered atlas labels layer",
        ),
        data_layer=dict(
            label="Data layer",
            tooltip="The downsampled brain data layer",
        ),
        atlas_points_layer=dict(
            label="Atlas landmarks",
            tooltip="Points layer for atlas landmarks (will be created)",
        ),
        data_points_layer=dict(
            label="Data landmarks",
            tooltip="Points layer for data landmarks (will be created)",
        ),
        smoothing=dict(
            value=0.0,
            min=0.0,
            max=100.0,
            step=0.1,
            label="TPS smoothing",
            tooltip="Thin-plate spline smoothing. 0=exact, higher=smoother",
        ),
        correction_mode=dict(
            label="Correction mode",
            tooltip="How to apply the correction",
        ),
        create_point_layers_button=dict(
            widget_type="PushButton",
            text="Create Point Layers",
        ),
        add_landmark_pair_button=dict(
            widget_type="PushButton",
            text="Add Landmark at Cursor (Both Layers)",
        ),
        compute_error_button=dict(
            widget_type="PushButton",
            text="Compute Registration Error",
        ),
        apply_correction_button=dict(
            widget_type="PushButton",
            text="Apply Correction",
        ),
        save_landmarks_button=dict(
            widget_type="PushButton",
            text="Save Landmarks",
        ),
        load_landmarks_button=dict(
            widget_type="PushButton",
            text="Load Landmarks",
        ),
        scrollable=True,
    )
    def widget(
        viewer: napari.Viewer,
        registration_folder: pathlib.Path = pathlib.Path.home(),
        load_registration_button=None,
        atlas_layer: Optional[napari.layers.Labels] = None,
        data_layer: Optional[napari.layers.Image] = None,
        atlas_points_layer: Optional[napari.layers.Points] = None,
        data_points_layer: Optional[napari.layers.Points] = None,
        smoothing: float = 0.0,
        correction_mode: CorrectionMode = CorrectionMode.POST_REGISTRATION,
        create_point_layers_button=None,
        add_landmark_pair_button=None,
        compute_error_button=None,
        apply_correction_button=None,
        save_landmarks_button=None,
        load_landmarks_button=None,
    ):
        """
        Landmark-based registration correction widget.

        Workflow:
        1. Load a completed brainreg registration
        2. Create point layers for landmarks
        3. Add corresponding points on atlas and data where errors are visible
        4. Apply correction to improve alignment

        Parameters
        ----------
        registration_folder : pathlib.Path
            Folder containing brainreg registration output.
        atlas_layer : napari.layers.Labels
            The registered atlas labels.
        data_layer : napari.layers.Image
            The downsampled brain data.
        atlas_points_layer : napari.layers.Points
            Points layer for atlas landmarks.
        data_points_layer : napari.layers.Points
            Points layer for data landmarks.
        smoothing : float
            TPS smoothing parameter. 0 = exact interpolation through points.
        correction_mode : CorrectionMode
            Whether to correct the atlas (post-registration) or warp the
            data for re-registration (pre-registration).
        """
        pass  # Widget body handled by button callbacks

    # Store state for the widget
    widget._registration_folder = None
    widget._metadata = None

    @widget.load_registration_button.changed.connect
    def load_registration(event=None):
        """Load a brainreg registration from the selected folder."""
        viewer = widget.viewer.value
        reg_folder = pathlib.Path(widget.registration_folder.value)

        # Check for brainreg.json
        meta_file = reg_folder / "brainreg.json"
        if not meta_file.exists():
            show_error(f"No brainreg.json found in {reg_folder}")
            return

        # Load metadata
        with open(meta_file) as f:
            widget._metadata = json.load(f)

        widget._registration_folder = reg_folder

        try:
            # Load registration layers
            boundaries, labels = add_registered_image_layers(
                viewer, registration_directory=reg_folder
            )

            # Also load downsampled data if available
            downsampled_path = reg_folder / "downsampled.tiff"
            if downsampled_path.exists():
                downsampled = tifffile.imread(downsampled_path)
                viewer.add_image(downsampled, name="Downsampled brain")

            show_info(f"Loaded registration from {reg_folder}")

            # Update layer dropdowns
            _refresh_layer_choices()

        except Exception as e:
            show_error(f"Failed to load registration: {e}")
            logger.exception("Failed to load registration")

    @widget.create_point_layers_button.changed.connect
    def create_point_layers(event=None):
        """Create point layers for landmark annotation."""
        viewer = widget.viewer.value

        # Check if layers already exist
        existing_names = [layer.name for layer in viewer.layers]

        if "Atlas landmarks" not in existing_names:
            atlas_points = viewer.add_points(
                name="Atlas landmarks",
                face_color="red",
                edge_color="white",
                size=10,
                symbol="cross",
                ndim=3,
            )
            atlas_points.mode = "add"
        else:
            show_info("Atlas landmarks layer already exists")

        if "Data landmarks" not in existing_names:
            data_points = viewer.add_points(
                name="Data landmarks",
                face_color="green",
                edge_color="white",
                size=10,
                symbol="cross",
                ndim=3,
            )
            data_points.mode = "add"
        else:
            show_info("Data landmarks layer already exists")

        _refresh_layer_choices()
        show_info(
            "Point layers created. Add corresponding landmarks:\n"
            "1. Select 'Atlas landmarks' and click on atlas\n"
            "2. Select 'Data landmarks' and click on data\n"
            "Points should be in the same order!"
        )

    @widget.add_landmark_pair_button.changed.connect
    def add_landmark_at_cursor(event=None):
        """Add a landmark at the current cursor position to both layers."""
        viewer = widget.viewer.value

        # Get current cursor position
        cursor_pos = viewer.cursor.position

        atlas_layer = _get_layer_by_name(viewer, "Atlas landmarks")
        data_layer = _get_layer_by_name(viewer, "Data landmarks")

        if atlas_layer is None or data_layer is None:
            show_error("Please create point layers first")
            return

        # Add point to both layers at current cursor position
        cursor_3d = np.array(cursor_pos[:3])

        atlas_layer.add(cursor_3d)
        data_layer.add(cursor_3d)

        show_info(
            f"Added landmark pair at {cursor_3d.astype(int)}. "
            f"Now drag points to their correct positions."
        )

    @widget.compute_error_button.changed.connect
    def compute_error(event=None):
        """Compute and display registration error from landmarks."""
        viewer = widget.viewer.value

        atlas_points = _get_points_from_layer(viewer, "Atlas landmarks")
        data_points = _get_points_from_layer(viewer, "Data landmarks")

        if atlas_points is None or data_points is None:
            show_error("Please add landmarks to both point layers")
            return

        if len(atlas_points) != len(data_points):
            show_error(
                f"Landmark count mismatch: {len(atlas_points)} atlas points, "
                f"{len(data_points)} data points. They must match!"
            )
            return

        if len(atlas_points) < 1:
            show_error("Please add at least one landmark pair")
            return

        error_stats = compute_registration_error(atlas_points, data_points)

        msg = (
            f"Registration Error Statistics ({error_stats['n_landmarks']} "
            f"landmarks):\n"
            f"  Mean error: {error_stats['mean_error']:.2f} voxels\n"
            f"  Median error: {error_stats['median_error']:.2f} voxels\n"
            f"  Max error: {error_stats['max_error']:.2f} voxels\n"
            f"  Std dev: {error_stats['std_error']:.2f} voxels"
        )
        show_info(msg)
        logger.info(msg)

    @widget.apply_correction_button.changed.connect
    def apply_correction(event=None):
        """Apply the landmark-based correction."""
        viewer = widget.viewer.value

        atlas_points = _get_points_from_layer(viewer, "Atlas landmarks")
        data_points = _get_points_from_layer(viewer, "Data landmarks")

        if atlas_points is None or data_points is None:
            show_error("Please add landmarks to both point layers")
            return

        if len(atlas_points) != len(data_points):
            show_error(
                f"Landmark count mismatch: {len(atlas_points)} atlas points, "
                f"{len(data_points)} data points"
            )
            return

        if len(atlas_points) < 4:
            show_error(
                "At least 4 landmark pairs required for thin-plate spline"
            )
            return

        smoothing = widget.smoothing.value
        mode = widget.correction_mode.value

        try:
            if mode == CorrectionMode.POST_REGISTRATION:
                # Correct the registered atlas
                atlas_layer = widget.atlas_layer.value
                if atlas_layer is None:
                    show_error("Please select an atlas layer")
                    return

                show_info("Applying correction to atlas... (this may take a moment)")

                corrected = correct_registered_atlas(
                    registered_atlas=atlas_layer.data,
                    atlas_landmarks=atlas_points,
                    data_landmarks=data_points,
                    smoothing=smoothing,
                )

                # Add corrected atlas as new layer
                viewer.add_labels(
                    corrected,
                    name=f"{atlas_layer.name} (corrected)",
                )

                # Optionally save
                if widget._registration_folder:
                    output_path = (
                        widget._registration_folder /
                        "registered_atlas_corrected.tiff"
                    )
                    tifffile.imwrite(output_path, corrected)
                    show_info(f"Corrected atlas saved to {output_path}")
                else:
                    show_info("Correction applied. Select output folder to save.")

            else:  # PRE_REGISTRATION
                # Warp the data for re-registration
                data_layer = widget.data_layer.value
                if data_layer is None:
                    show_error("Please select a data layer")
                    return

                show_info("Warping data... (this may take a moment)")

                warped = warp_data_for_registration(
                    brain_data=data_layer.data,
                    data_landmarks=data_points,
                    atlas_landmarks=atlas_points,
                    smoothing=smoothing,
                )

                # Add warped data as new layer
                viewer.add_image(
                    warped,
                    name=f"{data_layer.name} (pre-warped)",
                )

                # Save for re-registration
                if widget._registration_folder:
                    output_path = (
                        widget._registration_folder /
                        "downsampled_prewarped.tiff"
                    )
                    tifffile.imwrite(output_path, warped)
                    show_info(
                        f"Pre-warped data saved to {output_path}. "
                        f"Use this as input for re-registration."
                    )

            show_info("Correction applied successfully!")

        except Exception as e:
            show_error(f"Correction failed: {e}")
            logger.exception("Correction failed")

    @widget.save_landmarks_button.changed.connect
    def save_landmarks_callback(event=None):
        """Save landmarks to a JSON file."""
        viewer = widget.viewer.value

        atlas_points = _get_points_from_layer(viewer, "Atlas landmarks")
        data_points = _get_points_from_layer(viewer, "Data landmarks")

        if atlas_points is None or data_points is None or len(atlas_points) == 0:
            show_error("Please add landmarks before saving")
            return

        if len(atlas_points) != len(data_points):
            show_warning(
                f"Warning: Landmark count mismatch "
                f"({len(atlas_points)} vs {len(data_points)})"
            )

        if widget._registration_folder:
            output_path = widget._registration_folder / "landmarks.json"
        else:
            output_path = pathlib.Path.home() / "landmarks.json"

        save_landmarks(
            output_path,
            atlas_points,
            data_points,
            metadata=widget._metadata,
        )
        show_info(f"Landmarks saved to {output_path}")

    @widget.load_landmarks_button.changed.connect
    def load_landmarks_callback(event=None):
        """Load landmarks from a JSON file."""
        viewer = widget.viewer.value

        if widget._registration_folder:
            input_path = widget._registration_folder / "landmarks.json"
        else:
            input_path = pathlib.Path.home() / "landmarks.json"

        if not input_path.exists():
            show_error(f"No landmarks file found at {input_path}")
            return

        try:
            atlas_landmarks, data_landmarks, _ = load_landmarks(input_path)

            # Get or create point layers
            atlas_layer = _get_layer_by_name(viewer, "Atlas landmarks")
            data_layer = _get_layer_by_name(viewer, "Data landmarks")

            if atlas_layer is None or data_layer is None:
                # Create layers first
                create_point_layers()
                atlas_layer = _get_layer_by_name(viewer, "Atlas landmarks")
                data_layer = _get_layer_by_name(viewer, "Data landmarks")

            # Set the points
            atlas_layer.data = atlas_landmarks
            data_layer.data = data_landmarks

            show_info(f"Loaded {len(atlas_landmarks)} landmarks from {input_path}")

        except Exception as e:
            show_error(f"Failed to load landmarks: {e}")
            logger.exception("Failed to load landmarks")

    def _refresh_layer_choices():
        """Refresh the layer dropdown choices."""
        viewer = widget.viewer.value

        # Update Labels layer choices
        labels_layers = [
            layer for layer in viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]
        widget.atlas_layer.choices = labels_layers

        # Update Image layer choices
        image_layers = [
            layer for layer in viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        widget.data_layer.choices = image_layers

        # Update Points layer choices
        points_layers = [
            layer for layer in viewer.layers
            if isinstance(layer, napari.layers.Points)
        ]
        widget.atlas_points_layer.choices = points_layers
        widget.data_points_layer.choices = points_layers

    def _get_layer_by_name(viewer, name):
        """Get a layer by name, or None if not found."""
        for layer in viewer.layers:
            if layer.name == name:
                return layer
        return None

    def _get_points_from_layer(viewer, layer_name):
        """Get points data from a named layer."""
        layer = _get_layer_by_name(viewer, layer_name)
        if layer is None or len(layer.data) == 0:
            return None
        return np.array(layer.data)

    # Connect layer changes to refresh
    @widget.viewer.changed.connect
    def on_viewer_changed(event=None):
        if widget.viewer.value is not None:
            widget.viewer.value.layers.events.changed.connect(
                _refresh_layer_choices
            )
            widget.viewer.value.layers.events.inserted.connect(
                _refresh_layer_choices
            )
            widget.viewer.value.layers.events.removed.connect(
                _refresh_layer_choices
            )

    # Add header
    widget.native.layout().insertWidget(
        0,
        header_widget(
            "brainreg",
            "Landmark-based Registration Correction",
            tutorial_file_name="tutorial-whole-brain-registration.html",
            citation_doi="https://doi.org/10.1038/s41598-021-04676-9",
            help_text=(
                "Correct registration errors using correspondence points.\n"
                "1. Load a brainreg registration\n"
                "2. Create point layers for landmarks\n"
                "3. Add matching points on atlas and data\n"
                "4. Apply correction"
            ),
        ),
    )

    scroll = QScrollArea()
    scroll.setWidget(widget._widget._qwidget)
    widget._widget._qwidget = scroll

    return widget
