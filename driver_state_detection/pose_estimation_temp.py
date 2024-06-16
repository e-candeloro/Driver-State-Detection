import cv2
import numpy as np
from face_geometry import PCF, get_metric_landmarks, procrustes_landmark_basis
from utils import rot_mat_to_euler


class HeadPoseEstimator:

    def __init__(self, camera_matrix=None, dist_coeffs=None, show_axis: bool = False):
        """
        Class for estimating the head pose using the image/frame, face mesh landmarks, and camera parameters.

        Attributes
        ----------
        show_axis : bool
            If set to True, shows the head pose axis projected from the nose keypoint and the face landmarks points
            used for pose estimation (default is False).
        camera_matrix : numpy array
            Camera matrix of the camera used to capture the image/frame.
        dist_coeffs : numpy array
            Distortion coefficients of the camera used to capture the image/frame.
        focal_length : float
            Focal length of the camera used to capture the image/frame.
        pcf_calculated : bool
            Flag indicating whether the Perspective Camera Frustum (PCF) has been calculated.
        model_lms_ids : list
            List of model landmark IDs used for pose estimation.
        NOSE_AXES_POINTS : numpy array
            Array of nose axes points used for drawing the nose axes on the frame.

        Methods
        -------
        get_pose(frame, landmarks, frame_size)
            Estimate the head pose using the provided frame, landmarks, and frame size.
        _get_model_lms_ids()
            Get the model landmark IDs used for pose estimation.
        _draw_nose_axes(frame, rvec, tvec, model_img_lms)
            Draw the nose axes on the frame.
        _get_camera_parameters(frame_size)
            Get the camera parameters for pose estimation.
        """

        self.show_axis = show_axis
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.focal_length = None

        self.pcf_calculated = False

        self.model_lms_ids = self._get_model_lms_ids()

        self.NOSE_AXES_POINTS = np.array(
            [[7, 0, 10], [0, 7, 6], [0, 0, 14]], dtype=float
        )

    @staticmethod
    def _get_model_lms_ids():
        JAW_LMS_NUMS = [61, 291, 199]
        model_lms_ids = JAW_LMS_NUMS + \
            [key for key, _ in procrustes_landmark_basis]
        model_lms_ids.sort()

        return model_lms_ids

    def get_pose(self, frame, landmarks, frame_size):
        """
        Estimate head pose using the head pose estimator object instantiated attribute

        Parameters
        ----------
        frame: numpy array
            Image/frame captured by the camera
        landmarks: numpy array
            mediapiep face mesh detected 478 landmarks of the head

        Returns
        --------
        - if successful: image_frame, yaw, pitch, roll  (tuple)
        - if unsuccessful: None,None,None,None (tuple)

        """

        rvec = None
        tvec = None
        model_img_lms = None
        eulers = None
        metric_lms = None

        if not self.pcf_calculated:
            self._get_camera_parameters(frame_size)

        model_img_lms = (
            np.clip(landmarks[self.model_lms_ids, :2], 0.0, 1.0) * frame_size
        )

        metric_lms = get_metric_landmarks(landmarks.T.copy(), self.pcf)[0].T

        model_metric_lms = metric_lms[self.model_lms_ids, :]

        (solve_pnp_success, rvec, tvec) = cv2.solvePnP(
            model_metric_lms,
            model_img_lms,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        """
        The OpenCV Solve PnP method computes the rotation and translation vectors with respect to the camera coordinate 
        system of the image_points referred to the 3d head model_points. It takes into account the camera matrix and
        the distortion coefficients.
        The method used is iterative (cv2.SOLVEPNP_ITERATIVE)
        An alternative method can be the cv2.SOLVEPNP_SQPNP
        """
        tvec = tvec.round(2)

        if solve_pnp_success:
            rvec, tvec = cv2.solvePnPRefineVVS(
                model_metric_lms,
                model_img_lms,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
            )

            rvec1 = np.array(
                [rvec[2, 0], rvec[0, 0], rvec[1, 0]]).reshape((3, 1))

            # cv2.Rodrigues: convert a rotation vector to a rotation matrix (also known as a Rodrigues rotation matrix)
            rmat, _ = cv2.Rodrigues(rvec1)

            eulers = rot_mat_to_euler(rmat).reshape((-1, 1))

            """
            We use the rotationMatrixToEulerAngles function to compute the euler angles (roll, pitch, yaw) from the
            Rotation Matrix. This function also checks if we have a gymbal lock.
            The angles are converted from radians to degrees 
            
            An alternative method to compute the euler angles is the following:
        
            P = np.hstack((Rmat,tvec)) -> computing the projection matrix
            euler_angles = -cv2.decomposeProjectionMatrix(P)[6] -> extracting euler angles for yaw pitch and roll from the projection matrix
            """

            self._draw_nose_axes(frame, rvec, tvec, model_img_lms)

            return frame, eulers[0], eulers[1], eulers[2]

        else:
            return None, None, None, None

    def _draw_nose_axes(self, frame, rvec, tvec, model_img_lms):
        (nose_axes_point2D, _) = cv2.projectPoints(
            self.NOSE_AXES_POINTS, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        nose = tuple(model_img_lms[0, :2].astype(int))

        nose_x = tuple(nose_axes_point2D[0, 0].astype(int))
        nose_y = tuple(nose_axes_point2D[1, 0].astype(int))
        nose_z = tuple(nose_axes_point2D[2, 0].astype(int))

        cv2.line(frame, nose, nose_x, (255, 0, 0), 2)
        cv2.line(frame, nose, nose_y, (0, 255, 0), 2)
        cv2.line(frame, nose, nose_z, (0, 0, 255), 2)

    def _get_camera_parameters(self, frame_size):
        fr_w = frame_size[0]
        fr_h = frame_size[1]
        if self.camera_matrix is None:
            fr_center = (fr_w // 2, fr_h // 2)
            focal_length = fr_w
            self.camera_matrix = np.array(
                [
                    [focal_length, 0, fr_center[0]],
                    [0, focal_length, fr_center[1]],
                    [0, 0, 1],
                ],
                dtype="double",
            )
            self.focal_length = focal_length
        else:
            self.focal_length = self.camera_matrix[0, 0]
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((5, 1))

        self.pcf = PCF(frame_height=fr_h, frame_width=fr_w,
                       fy=self.focal_length)

        self.pcf_calculated = True
