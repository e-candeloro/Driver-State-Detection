# Camera Calibration Instructions

1. Create a folder named `calib_photos` inside the `camera_calibration` folder.
2. Print without any border adaptation, in an A4 paper sheet, the chessboard in the `pattern.png` file.
3. Attach the chessboard paper sheet to a planar/flat rigid surface, like a thick carboard piece or a clipboard.
4. With the desired camera/webcam, shoot various photos (20+) of the chessboard, with various angles.
5. Transfer all the photos to the `calib_photos` folder.
6. Ensure the `cameracalib.py` script has the correct path for the `calib_photos` folder.
7. The valid chessboard photos are visualized, after skipping them pressing a button, the camera coefficients are computed and printed out (may require some time).
8. Copy the camera parameters and save them to a file for late usage.

For further explanations, follow [this guide](https://learnopencv.com/camera-calibration-using-opencv/).

## Example
Camera parameters and distorsion coefficients in Python, initialized as numpy arrays

    camera_matrix = np.array([
        [899.12150372, 0., 644.26261492],
        [0., 899.45280671, 372.28009436],
        [0, 0,  1]
        ],
        dtype="double")

    dist_coeffs = np.array([
        [-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]
        ],
        dtype="double")