import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Driver State Detection")

    # selection the camera number, default is 0 (webcam)
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        metavar="",
        help="Camera number, default is 0 (webcam)",
    )

    parser.add_argument(
        "--camera_params",
        type=str,
        help="Path to the camera parameters file (JSON or YAML).",
    )

    # visualisation parameters
    parser.add_argument(
        "--show_fps",
        type=bool,
        default=True,
        metavar="",
        help="Show the actual FPS of the capture stream, default is true",
    )
    parser.add_argument(
        "--show_proc_time",
        type=bool,
        default=True,
        metavar="",
        help="Show the processing time for a single frame, default is true",
    )
    parser.add_argument(
        "--show_eye_proc",
        type=bool,
        default=False,
        metavar="",
        help="Show the eyes processing, deafult is false",
    )
    parser.add_argument(
        "--show_axis",
        type=bool,
        default=True,
        metavar="",
        help="Show the head pose axis, default is true",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        metavar="",
        help="Prints additional info, default is false",
    )

    # Attention Scorer parameters (EAR, Gaze Score, Pose)
    parser.add_argument(
        "--smooth_factor",
        type=float,
        default=0.5,
        metavar="",
        help="Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5",
    )
    parser.add_argument(
        "--ear_thresh",
        type=float,
        default=0.15,
        metavar="",
        help="Sets the EAR threshold for the Attention Scorer, default is 0.15",
    )
    parser.add_argument(
        "--ear_time_thresh",
        type=float,
        default=2,
        metavar="",
        help="Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds",
    )
    parser.add_argument(
        "--gaze_thresh",
        type=float,
        default=0.015,
        metavar="",
        help="Sets the Gaze Score threshold for the Attention Scorer, default is 0.2",
    )
    parser.add_argument(
        "--gaze_time_thresh",
        type=float,
        default=2,
        metavar="",
        help="Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2. seconds",
    )
    parser.add_argument(
        "--pitch_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees",
    )
    parser.add_argument(
        "--yaw_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees",
    )
    parser.add_argument(
        "--roll_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees",
    )
    parser.add_argument(
        "--pose_time_thresh",
        type=float,
        default=2.5,
        metavar="",
        help="Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds",
    )

    # parse the arguments and store them in the args variable dictionary
    args, _ = parser.parse_known_args()

    return args
