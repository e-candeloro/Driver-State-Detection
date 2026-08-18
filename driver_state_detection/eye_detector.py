import cv2
import numpy as np
from numpy import linalg as LA

from driver_state_detection.utils import resize


class EyeDetector:
    """Compute pixel-correct EAR and eye-width-normalized gaze from landmarks."""

    def __init__(self, show_processing: bool = False, preview_scale=300.0):
        self.show_processing = show_processing
        self.preview_scale = preview_scale
        self.EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]
        self.LEFT_IRIS_NUM = 468
        self.RIGHT_IRIS_NUM = 473

    @staticmethod
    def _calc_EAR_eye(eye_pts):
        """Return one eye's opening-to-width ratio, or ``None`` if collapsed."""
        eye_width = LA.norm(eye_pts[0] - eye_pts[1])
        if eye_width <= np.finfo(float).eps:
            return None
        ear_eye = (
            LA.norm(eye_pts[2] - eye_pts[3]) + LA.norm(eye_pts[4] - eye_pts[5])
        ) / (2 * eye_width)
        return ear_eye

    def show_eye_keypoints(self, color_frame, landmarks, frame_size):
        """
        Shows eyes keypoints found in the face, drawing red circles in their position in the frame/image

        Parameters
        ----------
        color_frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: landmarks: numpy array
            List of 478 mediapipe keypoints of the face
        """

        cv2.circle(
            color_frame,
            (landmarks[self.LEFT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
            3,
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.circle(
            color_frame,
            (landmarks[self.RIGHT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
            3,
            (255, 255, 255),
            cv2.FILLED,
        )

        for n in self.EYES_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def get_EAR(self, landmarks, frame_size):
        """Return mean Eye Aspect Ratio using ``frame_size=(width, height)``."""
        eye_pts_l = np.zeros(shape=(6, 2))
        eye_pts_r = eye_pts_l.copy()

        # Pixel coordinates prevent aspect ratio from distorting Euclidean distances.
        for i in range(len(self.EYES_LMS_NUMS) // 2):
            eye_pts_l[i] = landmarks[self.EYES_LMS_NUMS[i], :2] * frame_size
            eye_pts_r[i] = landmarks[self.EYES_LMS_NUMS[i + 6], :2] * frame_size

        ear_left = self._calc_EAR_eye(eye_pts_l)
        ear_right = self._calc_EAR_eye(eye_pts_r)

        ear_scores = [score for score in (ear_left, ear_right) if score is not None]
        if not ear_scores:
            return None
        ear_avg = float(np.mean(ear_scores))

        return ear_avg

    @staticmethod
    def _calc_1eye_score(landmarks, eye_lms_nums, eye_iris_num, frame_size, frame):
        """Return eye-width-normalized gaze and an optional debug eye crop.

        ``frame=None`` skips crop extraction. Degenerate eye width returns
        ``(None, None)`` instead of producing a non-finite score.
        """
        iris = landmarks[eye_iris_num, :2] * frame_size

        eye_x_min = landmarks[eye_lms_nums, 0].min()
        eye_y_min = landmarks[eye_lms_nums, 1].min()
        eye_x_max = landmarks[eye_lms_nums, 0].max()
        eye_y_max = landmarks[eye_lms_nums, 1].max()

        eye_center = (
            np.array(((eye_x_min + eye_x_max) / 2, (eye_y_min + eye_y_max) / 2))
            * frame_size
        )

        eye_width = (eye_x_max - eye_x_min) * frame_size[0]
        if eye_width <= np.finfo(float).eps:
            return None, None
        eye_gaze_score = LA.norm(iris - eye_center) / eye_width

        eye_x_min_frame = int(eye_x_min * frame_size[0])
        eye_y_min_frame = int(eye_y_min * frame_size[1])
        eye_x_max_frame = int(eye_x_max * frame_size[0])
        eye_y_max_frame = int(eye_y_max * frame_size[1])

        eye = None
        if frame is not None:
            eye = frame[
                eye_y_min_frame:eye_y_max_frame, eye_x_min_frame:eye_x_max_frame
            ]

        return eye_gaze_score, eye

    def get_Gaze_Score(self, frame, landmarks, frame_size):
        """Return mean iris displacement divided by eye width for both eyes."""

        left_gaze_score, left_eye = self._calc_1eye_score(
            landmarks,
            self.EYES_LMS_NUMS[:6],
            self.LEFT_IRIS_NUM,
            frame_size,
            frame if self.show_processing else None,
        )
        right_gaze_score, right_eye = self._calc_1eye_score(
            landmarks,
            self.EYES_LMS_NUMS[6:],
            self.RIGHT_IRIS_NUM,
            frame_size,
            frame if self.show_processing else None,
        )

        gaze_scores = [
            score for score in (left_gaze_score, right_gaze_score) if score is not None
        ]
        if not gaze_scores:
            return None
        avg_gaze_score = float(np.mean(gaze_scores))

        if (
            self.show_processing
            and left_eye is not None
            and right_eye is not None
            and left_eye.size
            and right_eye.size
        ):
            left_eye = resize(left_eye, self.preview_scale)
            right_eye = resize(right_eye, self.preview_scale)
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)

        return avg_gaze_score
