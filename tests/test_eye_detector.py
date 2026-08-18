import numpy as np
import pytest

from driver_state_detection.eye_detector import EyeDetector


def make_landmarks(frame_size, offset_x=0):
    landmarks = np.zeros((478, 3), dtype=float)
    detector = EyeDetector()
    eye_points = [
        [(100, 120), (140, 120), (110, 115), (110, 125), (130, 115), (130, 125)],
        [(200, 120), (240, 120), (210, 115), (210, 125), (230, 115), (230, 125)],
    ]
    for indices, points in zip(
        (detector.EYES_LMS_NUMS[:6], detector.EYES_LMS_NUMS[6:]), eye_points
    ):
        for index, (x, y) in zip(indices, points):
            landmarks[index, :2] = ((x + offset_x) / frame_size[0], y / frame_size[1])
    landmarks[detector.LEFT_IRIS_NUM, :2] = (
        (125 + offset_x) / frame_size[0],
        120 / frame_size[1],
    )
    landmarks[detector.RIGHT_IRIS_NUM, :2] = (
        (225 + offset_x) / frame_size[0],
        120 / frame_size[1],
    )
    return landmarks


def test_eye_scores_are_invariant_to_horizontal_translation():
    detector = EyeDetector()
    frame_size = (640, 480)
    original = make_landmarks(frame_size)
    translated = make_landmarks(frame_size, offset_x=100)

    assert detector.get_EAR(original, frame_size) == pytest.approx(
        detector.get_EAR(translated, frame_size)
    )
    assert detector.get_Gaze_Score(None, original, frame_size) == pytest.approx(
        detector.get_Gaze_Score(None, translated, frame_size)
    )


def test_eye_scores_are_invariant_to_frame_dimensions():
    detector = EyeDetector()
    small_size = (640, 480)
    large_size = (1280, 720)

    small = make_landmarks(small_size)
    large = make_landmarks(large_size)

    assert detector.get_EAR(small, small_size) == pytest.approx(
        detector.get_EAR(large, large_size)
    )
    assert detector.get_Gaze_Score(None, small, small_size) == pytest.approx(
        detector.get_Gaze_Score(None, large, large_size)
    )
