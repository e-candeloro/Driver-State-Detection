import pytest

from driver_state_detection.parser import (
    DEFAULT_DECAY_FACTOR,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_PERCLOS_THRESHOLD,
    DEFAULT_ROLL_BAR_LIMIT,
    get_args,
)


def test_boolean_options_can_be_disabled():
    args = get_args(["--no-show-fps", "--no-show-axis", "--verbose"])

    assert not args.show_fps
    assert not args.show_axis
    assert args.verbose


def test_unknown_option_is_rejected():
    with pytest.raises(SystemExit):
        get_args(["--ear-time-tresh", "5"])


def test_runtime_defaults_are_exposed():
    args = get_args([])

    assert args.min_detection_confidence == DEFAULT_MIN_DETECTION_CONFIDENCE
    assert args.perclos_thresh == DEFAULT_PERCLOS_THRESHOLD
    assert args.decay_factor == DEFAULT_DECAY_FACTOR
    assert args.roll_bar_limit == DEFAULT_ROLL_BAR_LIMIT


def test_custom_scoring_and_overlay_options():
    args = get_args(
        [
            "--perclos-thresh",
            "0.3",
            "--perclos-window",
            "30",
            "--decay-factor",
            "0.8",
            "--yaw-bar-limit",
            "75",
        ]
    )

    assert args.perclos_thresh == 0.3
    assert args.perclos_window == 30
    assert args.decay_factor == 0.8
    assert args.yaw_bar_limit == 75


@pytest.mark.parametrize(
    "arguments",
    [
        ["--ear-thresh", "0"],
        ["--pose-time-thresh", "-1"],
        ["--perclos-thresh", "1.1"],
        ["--decay-factor", "nan"],
        ["--min-detection-confidence", "inf"],
        ["--max-faces", "0"],
        ["--roll-thresh", "30", "--roll-bar-limit", "20"],
    ],
)
def test_invalid_numeric_options_are_rejected(arguments):
    with pytest.raises(SystemExit):
        get_args(arguments)
