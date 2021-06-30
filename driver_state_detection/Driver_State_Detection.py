import math
import time

import cv2
import dlib
import numpy as np
from numpy import linalg as LA
from numpy.lib.twodim_base import eye

camera_matrix = np.array(
    [[1.09520943e+03, 0.00000000e+00, 9.80688063e+02],
     [0.00000000e+00, 1.10470495e+03, 5.42055897e+02],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype="double")
# camera matrix obtained from the camera calibration script, using a 9x6 chessboard

dist_coeffs = np.array(
    [[1.41401053e-01, - 2.12991544e-01, - 8.88887657e-04,  1.03893066e-04,
      9.54437692e-02]], dtype="double")
# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard


def resize(frame, scale_percent):
    """
    Resize the image maintaining the aspect ratio
    :param frame: opencv image/frame
    :param scale_percent: int
        scale factor for resizing the image
    :return:
    resized: rescaled opencv image/frame
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    return resized


def get_face_area(face):
    """
    Computes the area of the bounding box ROI of the face detected by the dlib face detector
    It's used to sort the detected faces by the box area

    :param face: dlib bounding box of a detected face in faces
    :return: area of the face bounding box
    """
    return (face.left() - face.right()) * (face.bottom() - face.top())


def show_keypoints(keypoints, frame):
    """
    Draw circles on the opencv frame over the face keypoints predicted by the dlib predictor

    :param keypoints: dlib iterable 68 keypoints object
    :param frame: opencv frame
    :return: frame
        Returns the frame with all the 68 dlib face keypoints drawn
    """
    for n in range(0, 68):  # per tutti i 68 keypoints stampa su frame la loro posizione
        x = keypoints.part(n).x
        y = keypoints.part(n).y
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        return frame


def midpoint(p1, p2):
    """
    Compute the midpoint between two dlib keypoints

    :param p1: dlib single keypoint
    :param p2: dlib single keypoint
    :return: array of x,y coordinated of the midpoint between p1 and p2
    """
    return np.array([int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)])


def get_array_keypoints(landmarks, dtype="int", verbose: bool = False):
    """
    Converts all the iterable dlib 68 face keypoint in a numpy array of shape 68,2

    :param landmarks: dlib iterable 68 keypoints object
    :param dtype: dtype desired in output
    :param verbose: if set to True, prints array of keypoints (default is False)
    :return: points_array
        Numpy array containing all the 68 keypoints (x,y) coordinates
        The shape is 68,2
    """
    points_array = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        points_array[i] = (landmarks.part(i).x, landmarks.part(i).y)

    if verbose:
        print(points_array)

    return points_array


def isRotationMatrix(R):
    """
    Checks if a matrix is a rotation matrix
    :param R: np.array matrix of 3 by 3
    :return: True or False
        Return True if a matrix is a rotation matrix, False if not
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    """
    Computes the Tait–Bryan Euler angles from a Rotation Matrix.
    Also checks if there is a gymbal lock and eventually use an alternative formula
    :param R: np.array
        3 x 3 Rotation matrix
    :return: (roll, pitch, yaw) tuple of float numbers
        Euler angles in radians
    """
    # Calculates Tait–Bryan Euler angles from a Rotation Matrix
    assert (isRotationMatrix(R))  # check if it's a Rmat

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:  # check if it's a gymbal lock situation
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])

    else:  # if in gymbal lock, use different formula for yaw, pitch roll
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def draw_pose_info(frame, img_point, point_proj, roll=None, pitch=None, yaw=None):
    """
    Draw 3d orthogonal axis given a frame, a point in the frame, the projection point array.
    Also prints the information about the roll, pitch and yaw if passed

    :param frame: opencv image/frame
    :param img_point: tuple
        x,y position in the image/frame for the 3d axis for the projection
    :param point_proj: np.array
        Projected point along 3 axis obtained from the cv2.projectPoints function
    :param roll: float, optional
    :param pitch: float, optional
    :param yaw: float, optional
    :return: frame: opencv image/frame
        Frame with 3d axis drawn and, optionally, the roll,pitch and yaw values drawn
    """
    frame = cv2.line(frame, img_point, tuple(
        point_proj[0].ravel().astype(int)), (255, 0, 0), 3)
    frame = cv2.line(frame, img_point, tuple(
        point_proj[1].ravel().astype(int)), (0, 255, 0), 3)
    frame = cv2.line(frame, img_point, tuple(
        point_proj[2].ravel().astype(int)), (0, 0, 255), 3)

    if roll is not None and pitch is not None and yaw is not None:
        cv2.putText(frame, "Roll:" + str(round(roll, 3)), (500, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Pitch:" + str(round(pitch, 3)), (500, 70),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Yaw:" + str(round(yaw, 3)), (500, 90),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


class Eye_Detector:

    def __init__(self, frame, landmarks, show_processing: bool = False):
        """
        Eye dector class that contains various method for eye aperture rate estimation and gaze score estimation

        Parameters
        ----------
        frame: opencv/numpy image array
            contains frame to be processed
        landmarks:
            list of landmarks detected with dlib face 68 keypoint detector 
        show_processing: bool
            If set to True, shows frame images during the processing in some steps (default is False)

        Methods
        ----------
        - show_eye_keypoints: shows eye keypoints in the frame/image
        - get_EAR: computes EAR average score for the two eyes of the face
        - get_ROI: finds the ROI (Region Of Interest) of the eye, given the keypoints
        - get_Gaze_Score: computes the Gaze_Score (normalized euclidean distance between center of eye and pupil)
            of the eyes of the face
        """
        self.keypoints = landmarks
        self.frame = frame
        self.show_processing = show_processing
        self.eye_width = None

    def show_eye_keypoints(self, color_frame):
        """
        Shows eyes keypoints found in the face, drawing red circles in their position in the frame/image
        :param color_frame: opencv frame/image
        """

        for n in range(36, 48):
            x = self.keypoints.part(n).x
            y = self.keypoints.part(n).y
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def get_EAR(self):
        """

        :return: ear_score
            EAR average score between the two eyes
            The EAR or Eye Aspect Ratio is computed as the eye opennes divided by the eye lenght
            Each eye has his scores and the two scores are averaged
        """
        pts = self.keypoints
        i = 0  # auxiliary counter
        # numpy array for storing the keypoints positions of the left eye
        eye_pts_l = np.zeros(shape=(6, 2))
        # numpy array for storing the keypoints positions of the right eye
        eye_pts_r = np.zeros(shape=(6, 2))

        for n in range(36, 42):  # the dlib keypoints from 36 to 42 are referring to the left eye
            point_l = pts.part(n)  # save the i-keypoint of the left eye
            point_r = pts.part(n + 6)  # save the i-keypoint of the right eye
            # array of x,y coordinates for the left eye reference point
            eye_pts_l[i] = [point_l.x, point_l.y]
            # array of x,y coordinates for the right eye reference point
            eye_pts_r[i] = [point_r.x, point_r.y]
            i += 1  # increasing the auxiliary counter

        def EAR_eye(eye_pts):
            """
            Computer the EAR score for a single eyes given it's keypoints
            :param eye_pts: numpy array of shape (6,2) containing the keypoints of an eye considering the dlib ordering
            :return: ear_eye
                EAR of the eye
            """
            ear_eye = (LA.norm(eye_pts[1] - eye_pts[5]) + LA.norm(
                eye_pts[2] - eye_pts[4])) / (2 * LA.norm(eye_pts[0] - eye_pts[3]))
            '''
            EAR is computed as the mean of two measures of eye opening (see dlib face keypoints for the eye)
            divided by the eye lenght
            '''
            return ear_eye

        ear_left = EAR_eye(eye_pts_l)  # computing the left eye EAR score
        ear_right = EAR_eye(eye_pts_r)  # computing the right eye EAR score

        # computing the average EAR score
        ear_avg = (ear_left + ear_right) / 2

        return ear_avg

    def get_ROI(self, left_corner_keypoint_num: int):
        """
        Get the ROI bounding box of the eye given one of it's dlib keypoint found in the face

        :param left_corner_keypoint_num: most left dlib keypoint of the eye
        :return: eye_roi
            Sub-frame of the eye region of the opencv frame/image
        """

        keypoints = self.keypoints
        kp_num = left_corner_keypoint_num

        eye_array = np.array(
            [(keypoints.part(kp_num).x, keypoints.part(kp_num).y),
             (keypoints.part(kp_num+1).x, keypoints.part(kp_num+1).y),
             (keypoints.part(kp_num+2).x, keypoints.part(kp_num+2).y),
             (keypoints.part(kp_num+3).x, keypoints.part(kp_num+3).y),
             (keypoints.part(kp_num+4).x, keypoints.part(kp_num+4).y),
             (keypoints.part(kp_num+5).x, keypoints.part(kp_num+5).y)], np.int32)

        min_x = np.min(eye_array[:, 0])
        max_x = np.max(eye_array[:, 0])
        min_y = np.min(eye_array[:, 1])
        max_y = np.max(eye_array[:, 1])

        eye_roi = self.frame[min_y-2:max_y+2, min_x-2:max_x+2]

        return eye_roi

    def get_Gaze_Score(self):
        """
        Computes the average Gaze Score for the eyes
        The Gaze Score is the mean of the l2 norm (euclidean distance) between the center point of the Eye ROI
        (eye bounding box) and the center of the eye-pupil

        :return: avg_gaze_score or None
            If successful, returns the float gaze score
            If unsuccessful, returns None

        """

        def get_gaze(eye_roi):
            """
            Computes the L2 norm between the center point of the Eye ROI
            (eye bounding box) and the center of the eye pupil
            :param eye_roi: float
            :return: (gaze_score, eye_roi): tuple
                tuple
            """

            eye_center = np.array(
                [(eye_roi.shape[1] // 2), (eye_roi.shape[0] // 2)])  # eye ROI center position
            gaze_score = None
            circles = None

            # a bilateral filter is applied for reducing noise and keeping eye details
            eye_roi = cv2.bilateralFilter(eye_roi, 4, 40, 40)

            circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 10,
                                       param1=90, param2=6, minRadius=1, maxRadius=9)
            # a Hough Transform is used to find the iris circle and his center (the pupil) on the grayscale eye_roi image with the contours drawn in white

            if circles is not None and len(circles) > 0:
                circles = np.uint16(np.around(circles))
                circle = circles[0][0, :]

                cv2.circle(
                    eye_roi, (circle[0], circle[1]), circle[2], (255, 255, 255), 1)
                cv2.circle(
                    eye_roi, (circle[0], circle[1]), 1, (255, 255, 255), -1)

                # pupil position is the first circle center found with the Hough Transform
                pupil_position = np.array([int(circle[0]), int(circle[1])])

                cv2.line(eye_roi, (eye_center[0], eye_center[1]), (
                    pupil_position[0], pupil_position[1]), (255, 255, 255), 1)

                gaze_score = LA.norm(
                    pupil_position - eye_center) / eye_center[0]
                # computes the L2 distance between the eye_center and the pupil position

            cv2.circle(eye_roi, (eye_center[0],
                                 eye_center[1]), 1, (0, 0, 0), -1)

            if gaze_score is not None:
                return gaze_score, eye_roi
            else:
                return None, None

        left_eye_ROI = self.get_ROI(36)  # computes the ROI for the left eye
        right_eye_ROI = self.get_ROI(42)  # computes the ROI for the right eye

        # computes the gaze scores for the eyes
        gaze_eye_left, left_eye = get_gaze(left_eye_ROI)
        gaze_eye_right, right_eye = get_gaze(right_eye_ROI)

        # if show_processing is True, shows the eyes ROI, eye center, pupil center and line distance
        if self.show_processing and (left_eye is not None) and (right_eye is not None):
            left_eye = resize(left_eye, 1000)
            right_eye = resize(right_eye, 1000)
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)

        if gaze_eye_left and gaze_eye_right:

            # computes the average gaze score for the 2 eyes
            avg_gaze_score = (gaze_eye_left + gaze_eye_left) / 2
            return avg_gaze_score

        else:
            return None


class Head_Pose_Estimator:

    def __init__(self, frame, landmarks, camera_matrix=None, dist_coeffs=None, verbose: bool = False):
        """
        Head Pose estimator class that contains the get_pose method for computing the three euler angles
        (roll, pitch, yaw) of the head. It uses the image/frame, the dlib detected landmarks of the head and,
        optionally the camera parameters

        Parameters
        ----------
        frame: opencv image array
            contains frame to be processed
        landmarks:
            list of landmarks detected with dlib face 68 keypoint detector
        verbose: bool
            If set to True, shows the head pose axis projected from the nose keypoint and the face landmarks points
            used for pose estimation (default is False)
        """

        self.verbose = verbose
        self.keypoints = landmarks  # dlib 68 landmarks
        self.frame = frame  # opencv image array

        self.axis = np.float32([[200, 0, 0],
                                [0, 200, 0],
                                [0, 0, 200]])
        # array that specify the length of the 3 projected axis from the nose

        if camera_matrix is None:
            # if no camera matrix is given, estimate camera parameters using picture size
            self.size = frame.shape
            self.focal_length = self.size[1]
            self.center = (self.size[1] / 2, self.size[0] / 2)
            self.camera_matrix = np.array(
                [[self.focal_length, 0, self.center[0]],
                 [0, self.focal_length, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )

        else:
            # take camera matrix
            self.camera_matrix = camera_matrix

        if dist_coeffs is None:  # if no distorsion coefficients are given, assume no lens distortion
            self.dist_coeffs = np.zeros((4, 1))
        else:
            # take camera distortion coefficients
            self.dist_coeffs = dist_coeffs

        # 3D Head model world space points (generic human head)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])
        # 2D Point position of dlib face keypoints used for pose estimation
        self.image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),  # Chin
            (landmarks.part(36).x, landmarks.part(
                36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(
                45).y),  # Right eye right corne
            (landmarks.part(48).x, landmarks.part(
                48).y),  # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(
                54).y)  # Right mouth corner
        ], dtype="double")

    def get_pose(self):
        """
        Estimate head pose using the head pose estimator object instantiated attribute

        Returns
        --------
        - if successful: image_frame, roll, pitch, yaw (tuple)
        - if unsuccessful: None,None,None,None (tuple)

        """

        (success, rvec, tvec) = cv2.solvePnP(self.model_points, self.image_points,
                                             self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        '''
        The OpenCV Solve PnP method computes the rotation and translation vectors with respect to the camera coordinate 
        system of the image_points referred to the 3d head model_points. It takes into account the camera matrix and
        the distortion coefficients.
        The method used is iterative (cv2.SOLVEPNP_ITERATIVE)
        An alternative method can be the cv2.SOLVEPNP_SQPNP
        '''

        if success:  # if the solvePnP succeed, compute the head pose, otherwise return None

            rvec, tvec = cv2.solvePnPRefineVVS(
                self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs, rvec, tvec)
            # this method is used to refine the rvec and tvec prediction

            # Head nose point in the image plane
            nose = (int(self.image_points[0][0]), int(self.image_points[0][1]))

            (nose_end_point2D, _) = cv2.projectPoints(
                self.axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            # this function computes the 3 projection axis from the nose point of the head, so we can use them to
            # show the head pose later

            Rmat = cv2.Rodrigues(rvec)[0]
            # using the Rodrigues formula, this functions computes the Rotation Matrix from the rotation vector

            roll, pitch, yaw = 180 * \
                (rotationMatrixToEulerAngles(Rmat) / np.pi)
            """
            We use the rotationMatrixToEulerAngles function to compute the euler angles (roll, pitch, yaw) from the
            Rotation Matrix. This function also checks if we have a gymbal lock.
            The angles are converted from radians to degrees 
            """

            """
            An alternative method to compute the euler angles is the following:
            
            P = np.hstack((Rmat,tvec)) -> computing the projection matrix
            euler_angles = -cv2.decomposeProjectionMatrix(P)[6] -> extracting euler angles for yaw pitch and roll from the projection matrix
            """

            if self.verbose:
                # print("Camera Matrix :\n {0}".format(self.camera_matrix))
                # print ("Rotation Vector:\n {0}".format(rvec))
                # print ("Translation Vector:\n {0}".format(tvec))
                # print("Roll:"+ str(roll) + " Pitch: " + str(pitch) + " Yaw: " + str(yaw))
                self.frame = draw_pose_info(
                    self.frame, nose, nose_end_point2D, roll, pitch, yaw)
                # draws 3d axis from the nose and to the computed projection points
                for point in self.image_points:
                    cv2.circle(self.frame, tuple(
                        point.ravel().astype(int)), 2, (0, 255, 255), -1)
                # draws the 6 keypoints used for the pose estimation

            return self.frame, roll, pitch, yaw

        else:
            return None, None, None, None


class Attention_Scorer:

    def __init__(self, capture_fps: int, ear_tresh, gaze_tresh, perclos_tresh=0.2, ear_time_tresh=4.0, pitch_tresh=35,
                 yaw_tresh=30, gaze_time_tresh=4.0, roll_tresh=None, pose_time_tresh=4.0, verbose=False):
        """
        Attention Scorer class that contains methods for estimating EAR,Gaze_Score,PERCLOS and Head Pose over time,
        with the given thresholds (time tresholds and value tresholds)

        Parameters
        ----------
        capture_fps: int
            Upper frame rate of video/capture stream considered

        ear_tresh: float or int
            EAR score value threshold (if the EAR score is less than this value, eyes are considered closed!)

        gaze_tresh: float or int
            Gaze Score value treshold (if the Gaze Score is more than this value, the gaze is considered not centered)

        perclos_tresh: float (ranges from 0 to 1)
            PERCLOS treshold that indicates the maximum time allowed in 60 seconds of eye closure
            (default is 0.2 -> 20% of 1 minute)

        ear_time_tresh: float or int
            Maximum time allowable for consecutive eye closure (given the EAR threshold considered)
            (default is 4.0 seconds)

        pitch_tresh: int
            Treshold of the pitch angle for considering the person distracted (not looking in front)
            (default is 35 degrees from the center position)

        yaw_tresh: int
            Treshold of the yaw angle for considering the person distracted/unconscious (not straight neck)
            (default is 30 degrees from the straight neck position)

        roll_tresh: int
            Treshold of the roll angle for considering the person distracted/unconscious (not straight neck)
            (default is None: not considered)

        pose_time_tresh: float or int
            Maximum time allowable for consecutive distracted head pose (given the pitch,yaw and roll thresholds)
            (default is 4.0 seconds)

        verbose: bool
            If set to True, print additional information about the scores (default is False)


        Methods
        ----------

        - eval_scores: used to evaluate the driver state of attention
        - get_PERCLOS: specifically used to evaluate the driver sleepiness
        """

        self.fps = capture_fps
        self.delta_time_frame = (1.0 / capture_fps)  # estimated frame time
        self.prev_time = 0  # auxiliary variable for the PERCLOS estimation function
        # default time period for PERCLOS (60 seconds)
        self.perclos_time_period = 60
        self.perclos_tresh = perclos_tresh

        # the time tresholds are divided for the estimated frame time
        # (that is a function passed parameter and so can vary)
        self.ear_tresh = ear_tresh
        self.ear_act_tresh = ear_time_tresh / self.delta_time_frame
        self.ear_counter = 0
        self.eye_closure_counter = 0

        self.gaze_tresh = gaze_tresh
        self.gaze_act_tresh = gaze_time_tresh / self.delta_time_frame
        self.gaze_counter = 0

        self.roll_tresh = roll_tresh
        self.pitch_tresh = pitch_tresh
        self.yaw_tresh = yaw_tresh
        self.pose_act_tresh = pose_time_tresh / self.delta_time_frame
        self.pose_counter = 0

        self.verbose = verbose

    def eval_scores(self, ear_score, gaze_score, head_roll, head_pitch, head_yaw):
        """
        :param ear_score: float
            EAR (Eye Aspect Ratio) score obtained from the driver eye aperture
        :param gaze_score: float
            Gaze Score obtained from the driver eye gaze
        :param head_roll: float
            Roll angle obtained from the driver head pose
        :param head_pitch: float
            Pitch angle obtained from the driver head pose
        :param head_yaw: float
            Yaw angle obtained from the driver head pose

        :return:
            Returns a tuple of boolean values that indicates the driver state of attention
            tuple: (asleep, looking_away, distracted)
        """
        # instantiating state of attention variables
        asleep = False
        looking_away = False
        distracted = False

        if self.ear_counter >= self.ear_act_tresh:  # check if the ear cumulative counter surpassed the threshold
            asleep = True

        if self.gaze_counter >= self.gaze_act_tresh:  # check if the gaze cumulative counter surpassed the threshold
            looking_away = True

        if self.pose_counter >= self.pose_act_tresh:  # check if the pose cumulative counter surpassed the threshold
            distracted = True

        '''
        The 3 if blocks that follow are written in a way that when we have a score that's over it's value threshold, 
        a respective score counter (ear counter, gaze counter, pose counter) is increased and can reach a given maximum 
        over time.
        When a score doesn't surpass a threshold, it is diminished and can go to a minimum of zero.
        
        Example:
        
        If the ear score of the eye of the driver surpasses the threshold for a SINGLE frame, the ear_counter is increased.
        If the ear score of the eye is surpassed for multiple frames, the ear_counter will be increased and will reach 
        a given maximum, then it won't increase but the "asleep" variable will be set to True.
        When the ear_score doesn't surpass the threshold, the ear_counter is decreased. If there are multiple frame
        where the score doesn't surpass the threshold, the ear_counter can reach the minimum of zero
        
        This way, we have a cumulative score for each of the controlled features (EAR, GAZE and HEAD POSE).
        If high score it's reached for a cumulative counter, this function will retain its value and will need a
        bit of "cool-down time" to reach zero again 
        '''
        if (ear_score is not None) and (ear_score <= self.ear_tresh):
            if not asleep:
                self.ear_counter += 1
        elif self.ear_counter > 0:
            self.ear_counter -= 1

        if (gaze_score is not None) and (gaze_score >= self.gaze_tresh):
            if not looking_away:
                self.gaze_counter += 1
        elif self.gaze_counter > 0:
            self.gaze_counter -= 1

        if ((self.roll_tresh is not None and head_roll is not None and head_roll > self.roll_tresh) or (
                head_pitch is not None and abs(head_pitch) > self.pitch_tresh) or (
                head_yaw is not None and abs(head_yaw) > self.yaw_tresh)):
            if not distracted:
                self.pose_counter += 1
        elif self.pose_counter > 0:
            self.pose_counter -= 1

        if self.verbose:  # print additional info if verbose is True
            print(
                f"ear counter:{self.ear_counter}/{self.ear_act_tresh}\ngaze counter:{self.gaze_counter}/{self.gaze_act_tresh}\npose counter:{self.pose_counter}/{self.pose_act_tresh}")
            print(
                f"eye closed:{asleep}\tlooking away:{looking_away}\tdistracted:{distracted}")

        return asleep, looking_away, distracted

    def get_PERCLOS(self, ear_score):
        """

        :param ear_score: float
            EAR (Eye Aspect Ratio) score obtained from the driver eye aperture
        :return:
            tuple:(tired, perclos_score)

            tired:
                is a boolean value indicating if the driver is tired or not
            perclos_score:
                is a float value indicating the PERCLOS score over a minute
                after a minute this scores resets itself to zero
        """

        delta = time.time() - self.prev_time  # set delta timer
        tired = False  # set default value for the tired state of the driver

        # if the ear_score is lower or equal than the threshold, increase the eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_tresh):
            self.eye_closure_counter += 1

        # compute the cumulative eye closure time
        closure_time = (self.eye_closure_counter * self.delta_time_frame)
        # compute the PERCLOS over a given time period
        perclos_score = (closure_time) / self.perclos_time_period

        if perclos_score >= self.perclos_tresh:  # if the PERCLOS score is higher than a threshold, tired = True
            tired = True

        if self.verbose:
            print(
                f"Closure Time:{closure_time}/{self.perclos_time_period}\nPERCLOS: {round(perclos_score, 3)}")

        if delta >= self.perclos_time_period:  # at every end of the given time period, reset the counter and the timer
            self.eye_closure_counter = 0
            self.prev_time = time.time()

        return tired, perclos_score


def main():

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)
    prev_time = 0  # previous time variable, used to set the FPS limit
    fps_lim = 11  # FPS upper limit value, needed for estimating the time for each frame and increasing performances
    time_lim = 1. / fps_lim  # time window for each frame taken by the webcam

    # instantiation of the dlib face detector object
    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor(
        "predictor/shape_predictor_68_face_landmarks.dat")  # instantiation of the dlib keypoint detector model
    '''
    the keypoint predictor is compiled in C++ and saved as a .dat inside the "predictor" folder in the project
    inside the folder there is also a useful face keypoint image map to understand the position and numnber of the
    various predicted face keypoints
    '''

    Scorer = Attention_Scorer(fps_lim, ear_tresh=0.15, ear_time_tresh=2, gaze_tresh=0.2,
                              gaze_time_tresh=2, pitch_tresh=35, yaw_tresh=28, pose_time_tresh=2.5, verbose=False)
    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    while True:  # infinite loop for webcam video capture

        delta_time = time.time() - prev_time  # delta time for FPS capping
        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
            prev_time = time.time()

            # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
            ctime = time.time()
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

                # instantiate the Eye detector and pose estimator objects
                Eye_det = Eye_Detector(gray, landmarks, show_processing=True)
                Head_pose = Head_Pose_Estimator(
                    frame, landmarks, verbose=True)

                # shows the eye keypoints (can be commented)
                Eye_det.show_eye_keypoints(frame)

                ear = Eye_det.get_EAR()  # compute the EAR score of the eyes
                # compute the PERCLOS score and state of tiredness
                tired, perclos_score = Scorer.get_PERCLOS(ear)
                gaze = Eye_det.get_Gaze_Score()  # compute the Gaze Score
                frame_det, roll, pitch, yaw = Head_pose.get_pose()  # compute the head pose

                if frame_det is not None:  # if the head pose estimation is successful, show the results
                    frame = frame_det

                if ear is not None:  # show the real-time EAR score
                    cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                if gaze is not None:  # show the real-time Gaze Score
                    cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                # show the real-time PERCLOS score
                cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                if tired:  # if the driver is tired, show and alert on screen
                    cv2.putText(frame, "TIRED!", (10, 280),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                asleep, looking_away, distracted = Scorer.eval_scores(
                    ear, gaze, roll, pitch, yaw)  # evaluate the scores for EAR, GAZE and HEAD POSE

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
