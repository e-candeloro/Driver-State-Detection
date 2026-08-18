import cv2
import numpy as np

from driver_state_detection.face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)
from driver_state_detection.utils import rot_mat_to_euler


class HeadPoseEstimator:
    """Estimate roll, pitch, and yaw from MediaPipe landmarks and camera intrinsics."""

    def __init__(
        self,
        camera_matrix=None,
        dist_coeffs=None,
        camera_image_size=None,
        show_axis: bool = False,
    ):
        self.NOSE_AXES_POINTS = np.array(
            [[7, 0, 10], [0, 7, 6], [0, 0, 14]], dtype=float
        )
        self.JAW_LMS_NUMS = [61, 291, 199]

        self.show_axis = show_axis
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_image_size = camera_image_size
        self.focal_length = None

        self.pcf_calculated = False

        self.model_lms_ids = self._get_model_lms_ids()

    def _get_model_lms_ids(self):
        """Combine stable jaw anchors with the weighted Procrustes basis."""
        model_lms_ids = self.JAW_LMS_NUMS + [
            key for key, _ in procrustes_landmark_basis
        ]
        model_lms_ids.sort()

        return model_lms_ids

    def get_pose(self, frame, landmarks, frame_size):
        """Return ``(frame, roll, pitch, yaw)`` in degrees, or four ``None`` values."""

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

        solve_pnp_success, rvec, tvec = cv2.solvePnP(
            model_metric_lms,
            model_img_lms,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if solve_pnp_success:
            rvec, tvec = cv2.solvePnPRefineVVS(
                model_metric_lms,
                model_img_lms,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
            )

            # Convert MediaPipe's metric axes into the Euler convention used by the UI.
            rvec1 = np.array([rvec[2, 0], rvec[0, 0], rvec[1, 0]]).reshape((3, 1))

            rmat, _ = cv2.Rodrigues(rvec1)

            eulers = rot_mat_to_euler(rmat).reshape((-1, 1))

            if self.show_axis:
                self._draw_nose_axes(frame, rvec, tvec, model_img_lms)

            return frame, float(eulers[0, 0]), float(eulers[1, 0]), float(eulers[2, 0])

        else:
            return None, None, None, None

    def _draw_nose_axes(self, frame, rvec, tvec, model_img_lms):
        """Draw x, y, and z nose axes in blue, green, and red (OpenCV BGR)."""
        nose_axes_point2D, _ = cv2.projectPoints(
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
        """Create or resolution-scale intrinsics and cache the resulting PCF."""
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
            if self.camera_image_size and tuple(frame_size) != tuple(
                self.camera_image_size
            ):
                scale_x = fr_w / self.camera_image_size[0]
                scale_y = fr_h / self.camera_image_size[1]
                self.camera_matrix = self.camera_matrix.copy()
                self.camera_matrix[0, 0] *= scale_x
                self.camera_matrix[0, 2] *= scale_x
                self.camera_matrix[1, 1] *= scale_y
                self.camera_matrix[1, 2] *= scale_y
            self.focal_length = self.camera_matrix[1, 1]
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((5, 1))

        self.pcf = PCF(frame_height=fr_h, frame_width=fr_w, fy=self.focal_length)

        self.pcf_calculated = True
