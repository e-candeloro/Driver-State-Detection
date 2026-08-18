import argparse
import math
from pathlib import Path

DEFAULT_CAMERA = 0
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_FACE_PRESENCE_CONFIDENCE = 0.5
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5
DEFAULT_MAX_FACES = 1

DEFAULT_EAR_THRESHOLD = 0.15
DEFAULT_EAR_TIME_THRESHOLD = 2.0
DEFAULT_GAZE_THRESHOLD = 0.2
DEFAULT_GAZE_TIME_THRESHOLD = 2.0
DEFAULT_PERCLOS_THRESHOLD = 0.2
DEFAULT_PERCLOS_WINDOW = 60.0
DEFAULT_PERCLOS_MIN_VALID_FRACTION = 0.8
DEFAULT_ROLL_THRESHOLD = 20.0
DEFAULT_PITCH_THRESHOLD = 20.0
DEFAULT_YAW_THRESHOLD = 20.0
DEFAULT_POSE_TIME_THRESHOLD = 2.5
DEFAULT_DECAY_FACTOR = 0.9

DEFAULT_ROLL_BAR_LIMIT = 45.0
DEFAULT_PITCH_BAR_LIMIT = 45.0
DEFAULT_YAW_BAR_LIMIT = 60.0
DEFAULT_EYE_PREVIEW_SCALE = 300.0


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a finite number")
    return number


def _positive_float(value):
    number = _finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _probability(value):
    number = _finite_float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _positive_probability(value):
    number = _probability(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def get_args(argv=None):
    """Parse application options from ``argv`` or the process command line."""
    parser = argparse.ArgumentParser(description="Driver State Detection")

    input_group = parser.add_argument_group("input and model")
    input_group.add_argument(
        "-c",
        "--camera",
        type=int,
        default=DEFAULT_CAMERA,
        help=f"Camera number (default: {DEFAULT_CAMERA})",
    )
    input_group.add_argument(
        "--camera-params",
        "--camera_params",
        type=Path,
        help="Path to a JSON camera parameters file",
    )
    input_group.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"MediaPipe Face Landmarker model (default: {DEFAULT_MODEL_PATH})",
    )

    detector_group = parser.add_argument_group("face detector")
    detector_group.add_argument(
        "--min-detection-confidence",
        type=_probability,
        default=DEFAULT_MIN_DETECTION_CONFIDENCE,
        help=f"Minimum face detection confidence (default: {DEFAULT_MIN_DETECTION_CONFIDENCE})",
    )
    detector_group.add_argument(
        "--min-face-presence-confidence",
        type=_probability,
        default=DEFAULT_MIN_FACE_PRESENCE_CONFIDENCE,
        help=f"Minimum face presence confidence (default: {DEFAULT_MIN_FACE_PRESENCE_CONFIDENCE})",
    )
    detector_group.add_argument(
        "--min-tracking-confidence",
        type=_probability,
        default=DEFAULT_MIN_TRACKING_CONFIDENCE,
        help=f"Minimum tracking confidence (default: {DEFAULT_MIN_TRACKING_CONFIDENCE})",
    )
    detector_group.add_argument(
        "--max-faces",
        type=_positive_int,
        default=DEFAULT_MAX_FACES,
        help=f"Maximum number of faces to detect (default: {DEFAULT_MAX_FACES})",
    )

    display_group = parser.add_argument_group("display")
    display_group.add_argument(
        "--show-dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the attention and pose dashboard (default: true)",
    )
    display_group.add_argument(
        "--show-fps",
        "--show_fps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show capture FPS (default: true)",
    )
    display_group.add_argument(
        "--show-proc-time",
        "--show_proc_time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show frame processing time (default: true)",
    )
    display_group.add_argument(
        "--show-eye-keypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show eye and iris landmarks (default: true)",
    )
    display_group.add_argument(
        "--show-eye-proc",
        "--show_eye_proc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show enlarged eye crops (default: false)",
    )
    display_group.add_argument(
        "--eye-preview-scale",
        type=_positive_float,
        default=DEFAULT_EYE_PREVIEW_SCALE,
        help=f"Eye-crop preview size in percent (default: {DEFAULT_EYE_PREVIEW_SCALE})",
    )
    display_group.add_argument(
        "--show-axis",
        "--show_axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the head pose axes (default: true)",
    )
    display_group.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print timing and configuration details (default: false)",
    )

    score_group = parser.add_argument_group("attention scoring")
    score_group.add_argument(
        "--ear-thresh",
        "--ear_thresh",
        type=_positive_float,
        default=DEFAULT_EAR_THRESHOLD,
        help=f"Closed-eye EAR threshold (default: {DEFAULT_EAR_THRESHOLD})",
    )
    score_group.add_argument(
        "--ear-time-thresh",
        "--ear_time_thresh",
        type=_positive_float,
        default=DEFAULT_EAR_TIME_THRESHOLD,
        help=f"Seconds of eye closure before an alert (default: {DEFAULT_EAR_TIME_THRESHOLD})",
    )
    score_group.add_argument(
        "--gaze-thresh",
        "--gaze_thresh",
        type=_positive_float,
        default=DEFAULT_GAZE_THRESHOLD,
        help=f"Off-center gaze threshold (default: {DEFAULT_GAZE_THRESHOLD})",
    )
    score_group.add_argument(
        "--gaze-time-thresh",
        "--gaze_time_thresh",
        type=_positive_float,
        default=DEFAULT_GAZE_TIME_THRESHOLD,
        help=f"Seconds of off-center gaze before an alert (default: {DEFAULT_GAZE_TIME_THRESHOLD})",
    )
    score_group.add_argument(
        "--perclos-thresh",
        type=_positive_probability,
        default=DEFAULT_PERCLOS_THRESHOLD,
        help=f"Closed-eye fraction that indicates tiredness (default: {DEFAULT_PERCLOS_THRESHOLD})",
    )
    score_group.add_argument(
        "--perclos-window",
        type=_positive_float,
        default=DEFAULT_PERCLOS_WINDOW,
        help=f"Rolling PERCLOS window in seconds (default: {DEFAULT_PERCLOS_WINDOW})",
    )
    score_group.add_argument(
        "--perclos-min-valid-fraction",
        type=_positive_probability,
        default=DEFAULT_PERCLOS_MIN_VALID_FRACTION,
        help=(
            "Required valid-data fraction before PERCLOS alerts "
            f"(default: {DEFAULT_PERCLOS_MIN_VALID_FRACTION})"
        ),
    )
    score_group.add_argument(
        "--pitch-thresh",
        "--pitch_thresh",
        type=_positive_float,
        default=DEFAULT_PITCH_THRESHOLD,
        help=f"Absolute pitch alert threshold in degrees (default: {DEFAULT_PITCH_THRESHOLD})",
    )
    score_group.add_argument(
        "--yaw-thresh",
        "--yaw_thresh",
        type=_positive_float,
        default=DEFAULT_YAW_THRESHOLD,
        help=f"Absolute yaw alert threshold in degrees (default: {DEFAULT_YAW_THRESHOLD})",
    )
    score_group.add_argument(
        "--roll-thresh",
        "--roll_thresh",
        type=_positive_float,
        default=DEFAULT_ROLL_THRESHOLD,
        help=f"Absolute roll alert threshold in degrees (default: {DEFAULT_ROLL_THRESHOLD})",
    )
    score_group.add_argument(
        "--pose-time-thresh",
        "--pose_time_thresh",
        type=_positive_float,
        default=DEFAULT_POSE_TIME_THRESHOLD,
        help=f"Seconds outside pose limits before an alert (default: {DEFAULT_POSE_TIME_THRESHOLD})",
    )
    score_group.add_argument(
        "--decay-factor",
        type=_probability,
        default=DEFAULT_DECAY_FACTOR,
        help=f"Per-second timer decay factor (default: {DEFAULT_DECAY_FACTOR})",
    )

    overlay_group = parser.add_argument_group("pose dashboard")
    overlay_group.add_argument(
        "--roll-bar-limit",
        type=_positive_float,
        default=DEFAULT_ROLL_BAR_LIMIT,
        help=f"Roll bar half-range in degrees (default: {DEFAULT_ROLL_BAR_LIMIT})",
    )
    overlay_group.add_argument(
        "--pitch-bar-limit",
        type=_positive_float,
        default=DEFAULT_PITCH_BAR_LIMIT,
        help=f"Pitch bar half-range in degrees (default: {DEFAULT_PITCH_BAR_LIMIT})",
    )
    overlay_group.add_argument(
        "--yaw-bar-limit",
        type=_positive_float,
        default=DEFAULT_YAW_BAR_LIMIT,
        help=f"Yaw bar half-range in degrees (default: {DEFAULT_YAW_BAR_LIMIT})",
    )

    args = parser.parse_args(argv)
    for name in ("roll", "pitch", "yaw"):
        threshold = getattr(args, f"{name}_thresh")
        bar_limit = getattr(args, f"{name}_bar_limit")
        if bar_limit < threshold:
            parser.error(f"--{name}-bar-limit must be at least --{name}-thresh")
    return args
