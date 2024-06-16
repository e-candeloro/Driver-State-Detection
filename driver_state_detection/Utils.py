import json

import cv2
import numpy as np


def load_camera_parameters(file_path):
    try:
        with open(file_path, "r") as file:
            if file_path.endswith(".json"):
                data = json.load(file)
            else:
                raise ValueError("Unsupported file format. Use JSON or YAML.")
            return (
                np.array(data["camera_matrix"], dtype="double"),
                np.array(data["dist_coeffs"], dtype="double"),
            )
    except Exception as e:
        print(f"Failed to load camera parameters: {e}")
        return None, None


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


def get_landmarks(lms):
    surface = 0
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) for point in lms0.landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0.0, 0] = 0.0
        landmarks[landmarks[:, 0] > 1.0, 0] = 1.0
        landmarks[landmarks[:, 1] < 0.0, 1] = 0.0
        landmarks[landmarks[:, 1] > 1.0, 1] = 1.0

        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks

    return biggest_face


def get_face_area(face):
    """
    Computes the area of the bounding box ROI of the face detected by the dlib face detector
    It's used to sort the detected faces by the box area

    :param face: dlib bounding box of a detected face in faces
    :return: area of the face bounding box
    """
    return abs((face.left() - face.right()) * (face.bottom() - face.top()))


def show_keypoints(keypoints, frame):
    """
    Draw circles on the opencv frame over the face keypoints predicted by the dlib predictor

    :param keypoints: dlib iterable 68 keypoints object
    :param frame: opencv frame
    :return: frame
        Returns the frame with all the 68 dlib face keypoints drawn
    """
    for n in range(0, 68):
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


def rot_mat_to_euler(rmat):
    """
    This function converts a rotation matrix into Euler angles. It first checks if the given matrix is a valid
    rotation matrix by comparing its calculated identity matrix to the identity matrix. If it is a valid rotation
    matrix, it checks for the presence of a gimbal lock situation. If there is no gimbal lock, it calculates the
    Euler angles using the arctan2 function. If there is a gimbal lock, it uses a different formula for yaw, pitch,
    and roll. The function then checks the signs of the angles and adjusts them accordingly. Finally, it returns the
    Euler angles in degrees, rounded to two decimal places.

    Parameters
    ----------
    rmat: A rotation matrix as a np.ndarray.

    Returns
    -------
    Euler angles in degrees as a np.ndarray.

    """
    rtr = np.transpose(rmat)
    r_identity = np.matmul(rtr, rmat)

    I = np.identity(3, dtype=rmat.dtype)
    if np.linalg.norm(r_identity - I) < 1e-6:
        sy = (rmat[:2, 0] ** 2).sum() ** 0.5
        singular = sy < 1e-6

        if not singular:  # check if it's a gimbal lock situation
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])

        else:  # if in gimbal lock, use different formula for yaw, pitch roll
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0

        if x > 0:
            x = np.pi - x
        else:
            x = -(np.pi + x)

        if z > 0:
            z = np.pi - z
        else:
            z = -(np.pi + z)

        return (np.array([x, y, z]) * 180.0 / np.pi).round(2)
    else:
        print("Isn't rotation matrix")


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
    frame = cv2.line(
        frame, img_point, tuple(point_proj[0].ravel().astype(int)), (255, 0, 0), 3
    )
    frame = cv2.line(
        frame, img_point, tuple(point_proj[1].ravel().astype(int)), (0, 255, 0), 3
    )
    frame = cv2.line(
        frame, img_point, tuple(point_proj[2].ravel().astype(int)), (0, 0, 255), 3
    )

    if roll is not None and pitch is not None and yaw is not None:
        cv2.putText(
            frame,
            "Roll:" + str(round(roll, 0)),
            (500, 50),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Pitch:" + str(round(pitch, 0)),
            (500, 70),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Yaw:" + str(round(yaw, 0)),
            (500, 90),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame
