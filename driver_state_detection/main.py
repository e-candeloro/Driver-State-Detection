import pprint
import time

from driver_state_detection.qt_compat import configure_qt_before_cv2_import

configure_qt_before_cv2_import()

import cv2
import mediapipe as mp

from driver_state_detection.attention_scorer import AttentionScorer
from driver_state_detection.eye_detector import EyeDetector
from driver_state_detection.overlays import draw_dashboard
from driver_state_detection.parser import get_args
from driver_state_detection.pose_estimation import HeadPoseEstimator
from driver_state_detection.qt_compat import configure_qt_fonts_after_cv2_import
from driver_state_detection.utils import get_landmarks, load_camera_parameters

configure_qt_fonts_after_cv2_import()


def main(argv=None):
    """Run webcam detection and own the lifetime of camera and GUI resources."""
    args = get_args(argv)
    if not args.model_path.is_file():
        raise SystemExit(
            f"Face Landmarker model not found at {args.model_path}. Run "
            "`driver-state-detection-download-model` first."
        )

    if not cv2.useOptimized():
        cv2.setUseOptimized(True)

    try:
        camera_matrix, dist_coeffs, camera_image_size = (
            load_camera_parameters(args.camera_params)
            if args.camera_params
            else (None, None, None)
        )
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Invalid camera parameters: {error}") from error

    if args.verbose:
        print("Arguments and parameters:")
        pprint.pp(vars(args), indent=4)
        print("Camera matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("Distortion coefficients:")
        pprint.pp(dist_coeffs, indent=4)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=args.max_faces,
        min_face_detection_confidence=args.min_detection_confidence,
        min_face_presence_confidence=args.min_face_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    eye_detector = EyeDetector(
        show_processing=args.show_eye_proc,
        preview_scale=args.eye_preview_scale,
    )
    head_pose = HeadPoseEstimator(
        show_axis=args.show_axis,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        camera_image_size=camera_image_size,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Cannot open camera {args.camera}")

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            now = time.perf_counter()
            scorer = AttentionScorer(
                t_now=now,
                ear_thresh=args.ear_thresh,
                gaze_time_thresh=args.gaze_time_thresh,
                roll_thresh=args.roll_thresh,
                pitch_thresh=args.pitch_thresh,
                yaw_thresh=args.yaw_thresh,
                ear_time_thresh=args.ear_time_thresh,
                gaze_thresh=args.gaze_thresh,
                perclos_thresh=args.perclos_thresh,
                perclos_time_period=args.perclos_window,
                perclos_min_valid_fraction=args.perclos_min_valid_fraction,
                pose_time_thresh=args.pose_time_thresh,
                decay_factor=args.decay_factor,
                verbose=args.verbose,
            )
            previous_frame_time = now
            stream_start = now
            previous_timestamp_ms = -1
            fps = 0.0
            while True:
                frame_time = time.perf_counter()
                elapsed = frame_time - previous_frame_time
                previous_frame_time = frame_time
                if elapsed > 0:
                    fps = 1.0 / elapsed

                received, frame = cap.read()
                if not received:
                    print("Cannot receive frame from camera or stream ended")
                    break
                if args.camera == 0:
                    frame = cv2.flip(frame, 1)

                processing_start = cv2.getTickCount()
                frame_size = (frame.shape[1], frame.shape[0])
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # MediaPipe VIDEO mode rejects duplicate timestamps.
                timestamp_ms = max(
                    previous_timestamp_ms + 1,
                    int((frame_time - stream_start) * 1000),
                )
                previous_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(mp_image, timestamp_ms)

                ear = gaze = perclos = None
                roll = pitch = yaw = None
                alerts = []
                face_detected = bool(result.face_landmarks)
                if result.face_landmarks:
                    landmarks = get_landmarks(result.face_landmarks)
                    if args.show_eye_keypoints:
                        eye_detector.show_eye_keypoints(frame, landmarks, frame_size)
                    ear = eye_detector.get_EAR(landmarks, frame_size)
                    gaze = eye_detector.get_Gaze_Score(frame, landmarks, frame_size)
                    tired, perclos = scorer.get_rolling_PERCLOS(frame_time, ear)
                    posed_frame, roll, pitch, yaw = head_pose.get_pose(
                        frame, landmarks, frame_size
                    )
                    if posed_frame is not None:
                        frame = posed_frame
                    asleep, looking_away, distracted = scorer.eval_scores(
                        frame_time, ear, gaze, roll, pitch, yaw
                    )

                    if tired:
                        alerts.append("TIRED")
                    if asleep:
                        alerts.append("ASLEEP")
                    if looking_away:
                        alerts.append("LOOKING AWAY")
                    if distracted:
                        alerts.append("DISTRACTED")
                else:
                    # Mark the gap as unknown instead of assigning it to reacquisition.
                    scorer.mark_face_missing(frame_time)

                processing_ms = (
                    (cv2.getTickCount() - processing_start) / cv2.getTickFrequency()
                ) * 1000
                draw_dashboard(
                    frame,
                    ear=ear,
                    gaze=gaze,
                    perclos=perclos,
                    perclos_ready=scorer.perclos_ready,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    ear_threshold=scorer.ear_thresh,
                    gaze_threshold=scorer.gaze_thresh,
                    perclos_threshold=scorer.perclos_thresh,
                    roll_threshold=scorer.roll_thresh,
                    pitch_threshold=scorer.pitch_thresh,
                    yaw_threshold=scorer.yaw_thresh,
                    roll_limit=args.roll_bar_limit,
                    pitch_limit=args.pitch_bar_limit,
                    yaw_limit=args.yaw_bar_limit,
                    alerts=alerts,
                    face_detected=face_detected,
                    fps=fps if args.show_fps else None,
                    processing_ms=processing_ms if args.show_proc_time else None,
                    show_panels=args.show_dashboard,
                )

                cv2.imshow("Press 'q' to terminate", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
