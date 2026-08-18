import numpy as np
import pytest

from driver_state_detection.overlays import (
    DANGER_COLOR,
    SAFE_COLOR,
    WARNING_COLOR,
    _signed_bar_geometry,
    _status_color,
    draw_dashboard,
    format_angle,
    format_percent,
    format_ratio,
)


def dashboard_kwargs():
    return {
        "ear": 0.23,
        "gaze": 0.12,
        "perclos": 0.18,
        "perclos_ready": True,
        "roll": -12.4,
        "pitch": 3.1,
        "yaw": 22.8,
        "ear_threshold": 0.15,
        "gaze_threshold": 0.2,
        "perclos_threshold": 0.2,
        "roll_threshold": 20.0,
        "pitch_threshold": 20.0,
        "yaw_threshold": 20.0,
        "roll_limit": 45.0,
        "pitch_limit": 45.0,
        "yaw_limit": 60.0,
        "alerts": (),
        "face_detected": True,
        "fps": 29.8,
        "processing_ms": 12.4,
    }


def test_signed_bar_is_symmetric_around_zero():
    positive = _signed_bar_geometry(10, 20, 100, 200, 40)
    negative = _signed_bar_geometry(-10, 20, 100, 200, 40)

    assert positive.center_x == 200
    assert positive.value_x - positive.center_x == positive.center_x - negative.value_x
    assert positive.positive_threshold_x - positive.center_x == (
        positive.center_x - positive.negative_threshold_x
    )


def test_signed_bar_clips_geometry_but_not_formatting():
    positive = _signed_bar_geometry(500, 20, 10, 100, 45)
    negative = _signed_bar_geometry(-500, 20, 10, 100, 45)

    assert positive.value_x == 110
    assert negative.value_x == 10
    assert format_angle(500) == "+500.0 deg"


def test_signed_bar_rejects_range_smaller_than_threshold():
    with pytest.raises(ValueError, match="include its threshold"):
        _signed_bar_geometry(5, 20, 10, 100, 10)


def test_metric_formatting_is_intuitive():
    assert format_ratio(0.234) == "0.23"
    assert format_percent(0.184) == "18%"
    assert format_angle(-12.36) == "-12.4 deg"
    assert format_angle(-0.001) == "+0.0 deg"
    assert format_ratio(None) == "--"
    assert format_angle(float("nan")) == "--"


def test_status_colors_follow_threshold_proximity():
    assert _status_color(5, 20) == SAFE_COLOR
    assert _status_color(18, 20) == WARNING_COLOR
    assert _status_color(21, 20) == DANGER_COLOR
    assert _status_color(0.1, 0.15, danger_below=True) == DANGER_COLOR
    assert _status_color(0.2, 0.2, danger_at_threshold=True) == DANGER_COLOR


@pytest.mark.parametrize("shape", [(240, 320), (480, 640), (720, 1280), (1080, 1920)])
def test_dashboard_renders_at_common_frame_sizes(shape):
    frame = np.zeros((*shape, 3), dtype=np.uint8)

    result = draw_dashboard(frame, **dashboard_kwargs())

    assert result is frame
    assert frame.shape == (*shape, 3)
    assert np.any(frame)


def test_dashboard_handles_missing_and_extreme_values():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    options = dashboard_kwargs()
    options.update(
        ear=None,
        gaze=None,
        perclos=None,
        perclos_ready=False,
        roll=-500,
        pitch=None,
        yaw=500,
        alerts=("ASLEEP", "DISTRACTED"),
        face_detected=False,
    )

    draw_dashboard(frame, **options)

    assert np.any(frame)


def test_dashboard_uses_compact_fallback_on_tiny_frames():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)

    draw_dashboard(frame, **dashboard_kwargs())

    assert np.any(frame[:60])
