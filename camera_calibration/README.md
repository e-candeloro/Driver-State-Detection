# Camera Calibration Instructions

1. Create a folder named `calib_photos` inside the `camera_calibration` folder.
2. Print without any border adaptation, in an A4 paper sheet, the chessboard in the `pattern.png` file.
3. Attach the chessboard paper sheet to a planar/flat rigid surface, like a thick carboard piece or a clipboard.
4. With the desired camera/webcam, shoot various photos (20+) of the chessboard, with various angles.
5. Transfer all the photos to the `calib_photos` folder.
6. Run `uv run python camera_calibration/cameracalib.py --output camera_calibration/camera_params.json` from the repository root.
7. Add `--show-corners` to briefly preview successful checkerboard detections.
8. Pass the generated file to the application with `--camera_params camera_calibration/camera_params.json`.

The checkerboard dimensions, minimum number of valid images, and preview duration can be changed with `--checkerboard-columns`, `--checkerboard-rows`, `--min-images`, and `--preview-ms`.

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
