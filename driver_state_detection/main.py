import time
import pprint

import cv2
import mediapipe as mp
import numpy as np

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters


def main():

    args = get_args()

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  # set OpenCV optimization to True
        except:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected"
            )

    if args.camera_params:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
    else:
        camera_matrix, dist_coeffs = None, None

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

    """instantiation of mediapipe face mesh model. This model give back 478 landmarks
    if the rifine_landmarks parameter is set to True. 468 landmarks for the face and
    the last 10 landmarks for the irises
    """
    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    # instantiation of the Eye Detector and Head Pose estimator objects
    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(
        show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
    )

    # timing variables
    prev_time = time.perf_counter()
    fps = 0.0  # Initial FPS value

    t_now = time.perf_counter()

    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores
    Scorer = AttScorer(
        t_now=t_now,
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose,
    )

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    # time.sleep(0.01)  # To prevent zero division error when calculating the FPS

    while True:  # infinite loop for webcam video capture

        # get current time in seconds
        t_now = time.perf_counter()

        # Calculate the time taken to process the previous frame
        elapsed_time = t_now - prev_time
        prev_time = t_now

        # calculate FPS
        if elapsed_time > 0:
            fps = np.round(1 / elapsed_time, 3)

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        # if the frame comes from webcam, flip it so it looks like a mirror.
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        # start the tick counter for computing the processing time for each frame
        e1 = cv2.getTickCount()

        # transform the BGR frame in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get the frame size
        frame_size = frame.shape[1], frame.shape[0]

        # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
        # gray = cv2.bilateralFilter(gray, 5, 10, 10)
        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        # find the faces using the face mesh model
        lms = Detector.process(gray).multi_face_landmarks

        if lms:  # process the frame only if at least a face is found
            # getting face landmarks and then take only the bounding box of the biggest face
            landmarks = get_landmarks(lms)

            # shows the eye keypoints (can be commented)
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            # compute the EAR score of the eyes
            ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

            # compute the PERCLOS score and state of tiredness
            tired, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)

            # compute the Gaze Score
            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size
            )

            # compute the head pose
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            # evaluate the scores for EAR, GAZE and HEAD POSE
            asleep, looking_away, distracted = Scorer.eval_scores(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
            )

            # if the head pose estimation is successful, show the results
            if frame_det is not None:
                frame = frame_det

            # show the real-time EAR score
            if ear is not None:
                cv2.putText(
                    frame,
                    "EAR:" + str(round(ear, 3)),
                    (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # show the real-time Gaze Score
            if gaze is not None:
                cv2.putText(
                    frame,
                    "Gaze Score:" + str(round(gaze, 3)),
                    (10, 80),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # show the real-time PERCLOS score
            cv2.putText(
                frame,
                "PERCLOS:" + str(round(perclos_score, 3)),
                (10, 110),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if roll is not None:
                cv2.putText(
                    frame,
                    "roll:" + str(roll.round(1)[0]),
                    (450, 40),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if pitch is not None:
                cv2.putText(
                    frame,
                    "pitch:" + str(pitch.round(1)[0]),
                    (450, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if yaw is not None:
                cv2.putText(
                    frame,
                    "yaw:" + str(yaw.round(1)[0]),
                    (450, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # if the driver is tired, show and alert on screen
            if tired:
                cv2.putText(
                    frame,
                    "TIRED!",
                    (10, 280),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # if the state of attention of the driver is not normal, show an alert on screen
            if asleep:
                cv2.putText(
                    frame,
                    "ASLEEP!",
                    (10, 300),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if looking_away:
                cv2.putText(
                    frame,
                    "LOOKING AWAY!",
                    (10, 320),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if distracted:
                cv2.putText(
                    frame,
                    "DISTRACTED!",
                    (10, 340),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        # stop the tick counter for computing the processing time for each frame
        e2 = cv2.getTickCount()
        # processign time in milliseconds
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        # print fps and processing time per frame on screen
        if args.show_fps:
            cv2.putText(
                frame,
                "FPS:" + str(round(fps)),
                (10, 400),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )
        if args.show_proc_time:
            cv2.putText(
                frame,
                "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + "ms",
                (10, 430),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )

        # show the frame on screen
        cv2.imshow("Press 'q' to terminate", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
