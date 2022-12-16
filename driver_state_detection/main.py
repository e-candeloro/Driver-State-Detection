import time
import argparse

import cv2
import dlib
import numpy as np

from Utils import get_face_area
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from Attention_Scorer_Module import AttentionScorer as AttScorer

# camera matrix obtained from the camera calibration script, using a 9x6 chessboard
camera_matrix = np.array(
    [[899.12150372, 0., 644.26261492],
     [0., 899.45280671, 372.28009436],
     [0, 0,  1]], dtype="double")

# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = np.array(
    [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")


def main():

    parser = argparse.ArgumentParser(description='Driver State Detection')
    parser.add_argument('-c', '--camera', type=int,
                        default=0, metavar='', help='Camera number, default is 0 (webcam)')
    parser.add_argument('-s', '--show', type=bool, default=False,
                        metavar='', help='Show the processing, deafult is false')
    parser.add_argument('-x', '--axis', type=bool, default=True,
                        metavar='', help='Show the head pose axis, default is true')
    # TODO: add option for choose if use camera matrix and dist coeffs

    args = parser.parse_args()

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)
    prev_time = 0  # previous time variable, used to set the FPS limit
    fps_lim = 11  # FPS upper limit value, needed for estimating the time for each frame and increasing performances
    time_lim = 1. / fps_lim  # time window for each frame taken by the webcam

    cv2.setUseOptimized(True)  # set OpenCV optimization to True

    # instantiation of the dlib face detector object
    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor(
        "predictor/shape_predictor_68_face_landmarks.dat")  # instantiation of the dlib keypoint detector model
    '''
    the keypoint predictor is compiled in C++ and saved as a .dat inside the "predictor" folder in the project
    inside the folder there is also a useful face keypoint image map to understand the position and numnber of the
    various predicted face keypoints
    '''

    # instantiation of the eye detector and pose estimator objects
    Eye_det = EyeDet(show_processing=args.show)

    Head_pose = HeadPoseEst(show_axis=args.axis)

    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores
    Scorer = AttScorer(fps_lim, ear_tresh=0.15, ear_time_tresh=2, gaze_tresh=0.2,
                       gaze_time_tresh=2, pitch_tresh=35, yaw_tresh=28, pose_time_tresh=2.5, verbose=False)

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    while True:  # infinite loop for webcam video capture

        delta_time = time.perf_counter() - prev_time  # delta time for FPS capping
        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

         # if the frame comes from webcam, flip it so it looks like a mirror.
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
            prev_time = time.perf_counter()

            # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
            ctime = time.perf_counter()
            fps = 1.0 / float(ctime - ptime)
            ptime = ctime
            cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)

            # transform the BGR frame in grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # apply a bilateral filter to lower noise but keep frame details
            gray = cv2.bilateralFilter(gray, 5, 10, 10)

            # find the faces using the dlib face detector
            faces = Detector(gray)

            if len(faces) > 0:  # process the frame only if at least a face is found

                # take only the bounding box of the biggest face
                faces = sorted(faces, key=get_face_area, reverse=True)
                driver_face = faces[0]

                # predict the 68 facial keypoints position
                landmarks = Predictor(gray, driver_face)

                # shows the eye keypoints (can be commented)
                Eye_det.show_eye_keypoints(
                    color_frame=frame, landmarks=landmarks)

                # compute the EAR score of the eyes
                ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

                # compute the PERCLOS score and state of tiredness
                tired, perclos_score = Scorer.get_PERCLOS(ear)

                # compute the Gaze Score
                gaze = Eye_det.get_Gaze_Score(
                    frame=gray, landmarks=landmarks)

                # compute the head pose
                frame_det, roll, pitch, yaw = Head_pose.get_pose(
                    frame=frame, landmarks=landmarks)

                # if the head pose estimation is successful, show the results
                if frame_det is not None:
                    frame = frame_det

                # show the real-time EAR score
                if ear is not None:
                    cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                # show the real-time Gaze Score
                if gaze is not None:
                    cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                # show the real-time PERCLOS score
                cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                # if the driver is tired, show and alert on screen
                if tired:
                    cv2.putText(frame, "TIRED!", (10, 280),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                # evaluate the scores for EAR, GAZE and HEAD POSE
                asleep, looking_away, distracted = Scorer.eval_scores(
                    ear, gaze, roll, pitch, yaw)

                # if the state of attention of the driver is not normal, show an alert on screen
                if asleep:
                    cv2.putText(frame, "ASLEEP!", (10, 300),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                if looking_away:
                    cv2.putText(frame, "LOOKING AWAY!", (10, 320),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                if distracted:
                    cv2.putText(frame, "DISTRACTED!", (10, 340),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Frame", frame)  # show the frame on screen

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
