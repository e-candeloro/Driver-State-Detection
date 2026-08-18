#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path

from driver_state_detection.qt_compat import configure_qt_before_cv2_import

configure_qt_before_cv2_import()

import cv2
import numpy as np

from driver_state_detection.qt_compat import configure_qt_fonts_after_cv2_import

configure_qt_fonts_after_cv2_import()

CHECKERBOARD = (6, 9)
MIN_CALIBRATION_IMAGES = 10
CORNER_PREVIEW_MS = 250
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(
    image_pattern: str,
    show_corners: bool = False,
    checkerboard=CHECKERBOARD,
    min_images=MIN_CALIBRATION_IMAGES,
    preview_ms=CORNER_PREVIEW_MS,
):
    """Calibrate from equal-sized checkerboard images with at least ten valid views."""
    image_paths = [Path(path) for path in sorted(glob.glob(image_pattern))]
    if not image_paths:
        raise ValueError(f"No calibration images matched {image_pattern!r}")

    object_template = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    object_template[:, :2] = np.mgrid[
        0 : checkerboard[0], 0 : checkerboard[1]
    ].T.reshape(-1, 2)
    object_points = []
    image_points = []
    image_size = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is not None and current_size != image_size:
            raise ValueError("All calibration images must have the same dimensions")
        image_size = current_size

        found, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue
        object_points.append(object_template.copy())
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        image_points.append(refined)

        if show_corners:
            cv2.drawChessboardCorners(image, checkerboard, refined, found)
            cv2.imshow("Calibration corners", image)
            cv2.waitKey(preview_ms)

    cv2.destroyAllWindows()
    if len(image_points) < min_images:
        raise ValueError(
            f"At least {min_images} checkerboard detections are required; "
            f"found {len(image_points)}"
        )

    error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return error, camera_matrix, dist_coeffs, image_size


def main(argv=None):
    """Write calibration matrix, distortion, image size, and error as JSON."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calibrate a camera from checkerboards"
    )
    parser.add_argument(
        "--images",
        default=str(script_dir / "calib_photos" / "*.jpg"),
        help="Calibration image glob",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    parser.add_argument("--show-corners", action="store_true")
    parser.add_argument("--checkerboard-columns", type=int, default=CHECKERBOARD[0])
    parser.add_argument("--checkerboard-rows", type=int, default=CHECKERBOARD[1])
    parser.add_argument("--min-images", type=int, default=MIN_CALIBRATION_IMAGES)
    parser.add_argument("--preview-ms", type=int, default=CORNER_PREVIEW_MS)
    args = parser.parse_args(argv)
    if (
        min(
            args.checkerboard_columns,
            args.checkerboard_rows,
            args.min_images,
            args.preview_ms,
        )
        <= 0
    ):
        parser.error(
            "checkerboard dimensions, min-images, and preview-ms must be positive"
        )

    try:
        error, camera_matrix, dist_coeffs, image_size = calibrate(
            args.images,
            args.show_corners,
            checkerboard=(args.checkerboard_columns, args.checkerboard_rows),
            min_images=args.min_images,
            preview_ms=args.preview_ms,
        )
    except ValueError as exception:
        parser.exit(1, f"Calibration failed: {exception}\n")

    result = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_size": list(image_size),
        "reprojection_error": error,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Calibration saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
