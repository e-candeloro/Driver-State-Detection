import json
from types import SimpleNamespace

import numpy as np
import pytest

from driver_state_detection.utils import get_landmarks, load_camera_parameters


def face(width, height):
    return [
        SimpleNamespace(x=0.1, y=0.1, z=0.0),
        SimpleNamespace(x=0.1 + width, y=0.1 + height, z=0.0),
    ]


def test_get_landmarks_selects_largest_face():
    landmarks = get_landmarks([face(0.4, 0.4), face(0.1, 0.1)])

    assert np.ptp(landmarks[:, 0]) == pytest.approx(0.4)


def test_load_camera_parameters_validates_and_returns_image_size(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(
        json.dumps(
            {
                "camera_matrix": [[100, 0, 50], [0, 101, 40], [0, 0, 1]],
                "dist_coeffs": [0, 0, 0, 0, 0],
                "image_size": [100, 80],
            }
        ),
        encoding="utf-8",
    )

    matrix, distortion, image_size = load_camera_parameters(path)

    assert matrix.shape == (3, 3)
    assert distortion.shape == (5, 1)
    assert image_size == (100, 80)


def test_load_camera_parameters_rejects_bad_matrix(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(
        json.dumps({"camera_matrix": [[1]], "dist_coeffs": [0, 0, 0, 0]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="camera_matrix"):
        load_camera_parameters(path)
