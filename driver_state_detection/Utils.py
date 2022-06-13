import numpy as np
import cv2


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

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:  # check if it's a gymbal lock situation
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])

    else:  # if in gymbal lock, use different formula for yaw, pitch roll
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
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
