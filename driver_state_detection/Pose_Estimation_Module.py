import cv2
import numpy as np

from Utils import rotationMatrixToEulerAngles, draw_pose_info


class HeadPoseEstimator:

    def __init__(self, camera_matrix=None, dist_coeffs=None, show_axis: bool = False):
        """
        Head Pose estimator class that contains the get_pose method for computing the three euler angles
        (roll, pitch, yaw) of the head. It uses the image/frame, the dlib detected landmarks of the head and,
        optionally the camera parameters

        Parameters
        ----------
        camera_matrix: numpy array
            Camera matrix of the camera used to capture the image/frame
        dist_coeffs: numpy array
            Distortion coefficients of the camera used to capture the image/frame
        show_axis: bool
            If set to True, shows the head pose axis projected from the nose keypoint and the face landmarks points
            used for pose estimation (default is False)
        """

        self.show_axis = show_axis
        self.camera_matrix = camera_matrix
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

    def get_pose(self, frame, landmarks, prev_landmarks=None, smoothing_factor=0.5):
        """
        Estimate head pose using the head pose estimator object instantiated attribute

        Parameters
        ----------
        frame: numpy array
            Image/frame captured by the camera
        landmarks: dlib.rectangle
            Dlib detected 68 landmarks of the head

        Returns
        --------
        - if successful: image_frame, yaw, pitch, roll  (tuple)
        - if unsuccessful: None,None,None,None (tuple)

        """
        self.keypoints = landmarks  # dlib 68 landmarks
        self.frame = frame  # opencv image array

        self.axis = np.float32([[200, 0, 0],
                                [0, 200, 0],
                                [0, 0, 200]])
        # array that specify the length of the 3 projected axis from the nose

        if self.camera_matrix is None:
            # if no camera matrix is given, estimate camera parameters using picture size
            self.size = frame.shape
            self.focal_length = self.size[1]
            self.center = (self.size[1] / 2, self.size[0] / 2)
            self.camera_matrix = np.array(
                [[self.focal_length, 0, self.center[0]],
                 [0, self.focal_length, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )

        if self.dist_coeffs is None:  # if no distorsion coefficients are given, assume no lens distortion
            self.dist_coeffs = np.zeros((4, 1))

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

        if prev_landmarks is not None:
            self.prev_image_points = np.array([
                (prev_landmarks.part(30).x, prev_landmarks.part(30).y),  # Nose tip
                (prev_landmarks.part(8).x, prev_landmarks.part(8).y),  # Chin
                (prev_landmarks.part(36).x, prev_landmarks.part(
                    36).y),  # Left eye left corner
                (prev_landmarks.part(45).x, prev_landmarks.part(
                    45).y),  # Right eye right corne
                (prev_landmarks.part(48).x, prev_landmarks.part(
                    48).y),  # Left Mouth corner
                (prev_landmarks.part(54).x, prev_landmarks.part(
                    54).y)  # Right mouth corner
            ], dtype="double")

        else:
            self.prev_image_points = self.image_points

        self.smoothing_factor = smoothing_factor

        # rotation matrix for flipping the z axis reference frame of the head so it is consistent with the camera ref. frame!
        self.R_flip = np.zeros((3, 3), dtype=np.float32)
        self.R_flip[0, 0] = 1.0
        self.R_flip[1, 1] = 1.0
        self.R_flip[2, 2] = -1.0  # flip z axis

        # smooth the image points using the previous image points
        self.image_points = self.smoothing_factor * self.prev_image_points + \
            (1 - self.smoothing_factor) * self.image_points

        # compute the pose of the head using the image points and the 3D model points
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

            # this function computes the 3 projection axis from the nose point of the head, so we can use them to
            # show the head pose later
            (nose_end_point2D, _) = cv2.projectPoints(
                self.axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)

            # using the Rodrigues formula, this functions computes the Rotation Matrix from the rotation vector
            Rmat = np.matrix(cv2.Rodrigues(rvec)[0])

            yaw, pitch, roll = rotationMatrixToEulerAngles(
                self.R_flip*Rmat) * 180/np.pi

            """
            We use the rotationMatrixToEulerAngles function to compute the euler angles (roll, pitch, yaw) from the
            Rotation Matrix. This function also checks if we have a gymbal lock.
            The angles are converted from radians to degrees 
            
            An alternative method to compute the euler angles is the following:
        
            P = np.hstack((Rmat,tvec)) -> computing the projection matrix
            euler_angles = -cv2.decomposeProjectionMatrix(P)[6] -> extracting euler angles for yaw pitch and roll from the projection matrix
            """

            if self.show_axis:
                self.frame = draw_pose_info(
                    self.frame, nose, nose_end_point2D, roll, pitch, yaw)
                # draws 3d axis from the nose and to the computed projection points
                for point in self.image_points:
                    cv2.circle(self.frame, tuple(
                        point.ravel().astype(int)), 2, (0, 255, 255), -1)
                # draws the 6 keypoints used for the pose estimation

            return self.frame, yaw, pitch, roll

        else:
            return None, None, None, None
