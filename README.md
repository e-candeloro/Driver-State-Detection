# Real Time Driver State Detection

Real Time webcam based driver attention state detection using Python, OpenCV and Dlib.

## How It Works?

This script search for the driver face, then use the dlib library to predict 68 facial keypoints.
With those keypoints, the following feature are computed:

- **EAR**: Eye Aspect Ratio, it's the normalized average eyes aperture and it's used to see how much the eyes are opened or closed
- **Gaze Score**: L2 Norm (Euclidean distance) between the center of the eye and the pupil, it's used to see if the driver is looking away or not
- **Head Pose**: Roll, Pitch and Yaw of the head of the driver. The angles are used to see if the driver is not looking straight ahead or doesn't have a straight head pose (probably unconscious)
- **PERCLOS**: PERcentage of CLOSure eye time, used to see how much time the eyes are closed in a minute. A threshold of 0.2 is used in this case (20% of a minute) and the EAR score is used to estimate when the eyes are closed.

The driver states can be classified as:
- **Normal**: no messages are printed
- **Tired**: when the PERCLOS score is > 0.2, a warning message is printed on screen
- **Asleep**: when the eyes are closed (EAR < closure_threshold) for a certain amount of time, a warning message is printed on screen
- **Looking Away**: when the gaze score is higher than a certain threshold for a certain amount of time, a warning message is printed on screen
- **Distracted**: when the head pose score is higher than a certain threshold for a certain amount of time, a warning message is printed on screen

## Demo

https://user-images.githubusercontent.com/67196406/121312501-bb571d00-c905-11eb-8d25-1cd28efc9110.mp4

### Gaze Score Estimation
The white line is the Euclidean (L2) distance between the black dot (center of the ROI of the eyes) and the white dot (estimated center of the iris/pupil)
Adaptive tresholding is used and then a detect contours is applied to enanche edges over the gray image. Then a Hough Transform is used to find the iris circle and his center (the pupil)

![Eye processing for gaze score estimation](https://user-images.githubusercontent.com/67196406/121316610-c8760b00-c909-11eb-9f25-3d600314285f.png)
![Gaze_Score estimation](https://user-images.githubusercontent.com/67196406/121316549-bc8a4900-c909-11eb-80cc-eb18155ce0f8.png)

L2 distance is used and normalized with the L2 eye width

https://user-images.githubusercontent.com/67196406/121316446-a1b7d480-c909-11eb-9bac-773b7994b05b.mp4

