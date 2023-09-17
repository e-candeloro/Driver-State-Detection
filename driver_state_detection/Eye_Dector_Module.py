import cv2
import numpy as np
from numpy import linalg as LA
from Utils import resize


EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]
LEFT_IRIS_NUM = 468
RIGHT_IRIS_NUM = 473

class EyeDetector:

    def __init__(self, show_processing: bool = False):
        """
        Eye dector class that contains various method for eye aperture rate estimation and gaze score estimation

        Parameters
        ----------
        show_processing: bool
            If set to True, shows frame images during the processing in some steps (default is False)

        Methods
        ----------
        - show_eye_keypoints: shows eye keypoints in the frame/image
        - get_EAR: computes EAR average score for the two eyes of the face
        - get_Gaze_Score: computes the Gaze_Score (normalized euclidean distance between center of eye and pupil)
            of the eyes of the face
        """

        self.keypoints = None
        self.frame = None
        self.show_processing = show_processing
        self.eye_width = None

    def show_eye_keypoints(self, color_frame, landmarks, frame_size):
        """
        Shows eyes keypoints found in the face, drawing red circles in their position in the frame/image

        Parameters
        ----------
        color_frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: list
            List of 68 dlib keypoints of the face
        """

    
        self.keypoints = landmarks

        for n in EYES_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def get_EAR(self, frame, landmarks):
        """
        Computes the average eye aperture rate of the face

        Parameters
        ----------
        frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: list
            List of 68 dlib keypoints of the face

        Returns
        -------- 
        ear_score: float
            EAR average score between the two eyes
            The EAR or Eye Aspect Ratio is computed as the eye opennes divided by the eye lenght
            Each eye has his scores and the two scores are averaged
        """

        self.keypoints = landmarks
        self.frame = frame
        pts = self.keypoints

        i = 0  # auxiliary counter
        # numpy array for storing the keypoints positions of the left eye
        eye_pts_l = np.zeros(shape=(6, 2))
        # numpy array for storing the keypoints positions of the right eye
        eye_pts_r = np.zeros(shape=(6, 2))

        for i in range(len(EYES_LMS_NUMS)//2):  # the dlib keypoints from 36 to 42 are referring to the left eye
            point_l = landmarks[EYES_LMS_NUMS[i]]
            point_r = landmarks[EYES_LMS_NUMS[i+6]]

            # array of x,y coordinates for the left eye reference point
            eye_pts_l[i] = [point_l[0], point_l[1]]
            # array of x,y coordinates for the right eye reference point
            eye_pts_r[i] = [point_r[0], point_r[1]]

        def EAR_eye(eye_pts):
            """
            Computer the EAR score for a single eyes given it's keypoints
            :param eye_pts: numpy array of shape (6,2) containing the keypoints of an eye considering the dlib ordering
            :return: ear_eye
                EAR of the eye
            """
            ear_eye = (LA.norm(eye_pts[2] - eye_pts[3]) + LA.norm(
                eye_pts[4] - eye_pts[5])) / (2 * LA.norm(eye_pts[0] - eye_pts[1]))
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

    def get_Gaze_Score(self, frame, landmarks, frame_size):
        """
        Computes the average Gaze Score for the eyes
        The Gaze Score is the mean of the l2 norm (euclidean distance) between the center point of the Eye ROI
        (eye bounding box) and the center of the eye-pupil

        Parameters
        ----------
        frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: list
            List of 68 dlib keypoints of the face

        Returns
        -------- 
        avg_gaze_score: float
            If successful, returns the float gaze score
            If unsuccessful, returns None

        """
        self.keypoints = landmarks
        self.frame = frame

        left_iris = landmarks[LEFT_IRIS_NUM, :2]
        right_iris = landmarks[RIGHT_IRIS_NUM, :2]

        left_eye_x_min = landmarks[EYES_LMS_NUMS[:6], 0].min()
        left_eye_y_min = landmarks[EYES_LMS_NUMS[:6], 1].min()
        left_eye_x_max = landmarks[EYES_LMS_NUMS[:6], 0].max()
        left_eye_y_max = landmarks[EYES_LMS_NUMS[:6], 1].max()

        right_eye_x_min = landmarks[EYES_LMS_NUMS[6:], 0].min()
        right_eye_y_min = landmarks[EYES_LMS_NUMS[6:], 1].min()
        right_eye_x_max = landmarks[EYES_LMS_NUMS[6:], 0].max()
        right_eye_y_max = landmarks[EYES_LMS_NUMS[6:], 1].max()
        
        left_eye_center = np.array(((left_eye_x_min+left_eye_x_max)/2,
                                    (left_eye_y_min+left_eye_y_max)/2))
        right_eye_center = np.array(((right_eye_x_min+right_eye_x_max)/2,
                                    (right_eye_y_min+right_eye_y_max)/2))
        
        left_gaze_score = LA.norm(left_iris - left_eye_center) / left_eye_center[0]
        right_gaze_score = LA.norm(right_iris - right_eye_center) / right_eye_center[0]

        # if show_processing is True, shows the eyes ROI, eye center, pupil center and line distance

        # computes the average gaze score for the 2 eyes
        avg_gaze_score = (left_gaze_score + right_gaze_score) / 2

        if self.show_processing and (left_eye is not None) and (right_eye is not None):
            left_eye_x_min_frame = int(left_eye_x_min * frame_size[0])
            left_eye_y_min_frame = int(left_eye_y_min * frame_size[1])
            left_eye_x_max_frame = int(left_eye_x_max * frame_size[0])
            left_eye_y_max_frame = int(left_eye_y_max * frame_size[1])
            right_eye_x_min_frame = int(right_eye_x_min * frame_size[0])
            right_eye_y_min_frame = int(right_eye_y_min * frame_size[1])
            right_eye_x_max_frame = int(right_eye_x_max * frame_size[0])
            right_eye_y_max_frame = int(right_eye_y_max * frame_size[1])

            left_eye = frame[left_eye_y_min_frame:left_eye_y_max_frame,
                             left_eye_x_min_frame:left_eye_x_max_frame]
            right_eye = frame[right_eye_y_min_frame:right_eye_y_max_frame,
                             right_eye_x_min_frame:right_eye_x_max_frame]

            left_eye = resize(left_eye, 1000)
            right_eye = resize(right_eye, 1000)
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)
        
        return avg_gaze_score
