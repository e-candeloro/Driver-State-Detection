# Real Time Driver State Detection

Real time, webcam based, driver attention state detection and monitoring using Python with the OpenCV and Dlib libraries.

![driver state detection demo](https://user-images.githubusercontent.com/67196406/173455413-ba95db40-6be5-4d64-9a1d-6c998854130e.gif)


**Note**:
This work is partially based on [this paper](https://www.researchgate.net/publication/327942674_Vision-Based_Driver%27s_Attention_Monitoring_System_for_Smart_Vehicles) for the scores and methods used.

## How Does It Work?

This script searches for the driver face, then use the dlib library to predict 68 facial keypoints.
The enumeration and location of all the face keypoints/landmarks can be seen [here](https://raw.githubusercontent.com/e-candeloro/Driver-State-Detection/master/predictor/Keypoint%20map%20example.png).

With those keypoints, the following scores are computed:

- **EAR**: Eye Aspect Ratio, it's the normalized average eyes aperture, and it's used to see how much the eyes are opened or closed
- **Gaze Score**: L2 Norm (Euclidean distance) between the center of the eye and the pupil, it's used to see if the driver is looking away or not
- **Head Pose**: Roll, Pitch and Yaw of the head of the driver. The angles are used to see if the driver is not looking straight ahead or doesn't have a straight head pose (is probably unconscious)
- **PERCLOS**: PERcentage of CLOSure eye time, used to see how much time the eyes are closed in a minute. A threshold of 0.2 is used in this case (20% of a minute) and the EAR score is used to estimate when the eyes are closed.

The driver states can be classified as:
- **Normal**: no messages are printed
- **Tired**: when the PERCLOS score is > 0.2, a warning message is printed on screen
- **Asleep**: when the eyes are closed (EAR < closure_threshold) for a certain amount of time, a warning message is printed on screen
- **Looking Away**: when the gaze score is higher than a certain threshold for a certain amount of time, a warning message is printed on screen
- **Distracted**: when the head pose score is higher than a certain threshold for a certain amount of time, a warning message is printed on screen

## Demo

https://user-images.githubusercontent.com/67196406/121312501-bb571d00-c905-11eb-8d25-1cd28efc9110.mp4

## The Scores Explained

### EAR
**Eye Aspect Ratio** is a normalized score that is useful to understand the rate of aperture of the eyes.
Using the dlib keypoints for each eye (six for each), the eye lenght and width are estimated and using this data the EAR score is computed as explained in the image below:
![EAR](https://user-images.githubusercontent.com/67196406/121489162-18210900-c9d4-11eb-9d2e-765f5ac42286.png)

**NOTE:** the average of the two eyes EAR score is computed


### Gaze Score Estimation
The gaze score gives information about how much the driver is looking away without turning his head.

To understand this, the distance between the eye center and the position of the pupil is computed. The result is then normalized by the eye width that can be different depending on the driver physionomy and distance from the camera.

The below image explains graphically how the Gaze Score for a single eye is computed:
![Gaze Score](https://user-images.githubusercontent.com/67196406/121489746-ab5a3e80-c9d4-11eb-8f33-d34afd0947b4.png)
**NOTE:** the average of the two eyes Gaze Score is computed

**Eye processing for gaze score:**

![Gaze_Score estimation](https://user-images.githubusercontent.com/67196406/121316549-bc8a4900-c909-11eb-80cc-eb18155ce0f8.png)

**Updated version of the processing for computing the gaze score**:

For the first version, an adaptive thresholding was used to aid the Hough transform for detecting the iris position.
In the updated version,after a Bilateral Filter to remove some noise, only the Hough transform is used and the ROI of the eyes has been reduced in size.

![Gaze_Score estimation v2](https://user-images.githubusercontent.com/67196406/123986317-7daa5900-d9c6-11eb-8860-bd23f8983bdc.png)


**Demo**

https://user-images.githubusercontent.com/67196406/121316446-a1b7d480-c909-11eb-9bac-773b7994b05b.mp4

The white line is the Euclidean (L2) distance between the black dot (center of the ROI of the eyes) and the white dot (estimated center of the iris/pupil).

### Head Pose Estimation
For the head pose estimation, a standard 3d head model in world coordinates was considered, in combination of the respective dlib keypoints in the image plane. 
In this way, using the solvePnP function of OpenCV, estimating the rotation and translation vector of the head in respect to the camera is possible.
Then the 3 Euler angles are computed.

The partial snippets of code used for this task can be found in [this article](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/).


## Installation

This projects runs on Python 3.9 with the following libraries:

- numpy
- OpenCV (opencv-python)
- Dlib and cmake

The Dlib predictor for face keypoints is already included in the "predictor" folder

### IMPORTANT: Dlib requirements and libraries installation

Dlib is a library that needs a C/C++ compiler installed and also the Cmake library. 
Please follow [this guide](http://dlib.net/compile.html) to install dlib propely on your machine.

If you have already all the prerequisites in your machine to install dlib and cmake you can use the requirements.txt file provided in the repository using:
    
    pip install -r requirements.txt
    
Or you can execute the following pip commands on terminal:

```
pip install numpy
pip install opencv-python
pip install cmake
pip install dlib --verbose
```

If you have difficulties installing dlib, it is suggested to use the .whl precompiled package available online.

## Why this project
This project was developed as part for a final group project for the course of [Computer Vision and Cognitive Systems](https://international.unimore.it/singleins.html?ID=295) done at the [University of Modena and Reggio Emilia](https://international.unimore.it/) in the second semester of the academic year 2020/2021.
Given the possible applications of Computer Vision, we wanted to focus mainly on the automotive field, developing a useful and potential life saving proof of concept project.
In fact, sadly, many fatal accidents happens [because of the driver distraction](https://www.nhtsa.gov/risky-driving/distracted-driving).

## License and Contacts

This project is freely available under the MIT license. You can use/modify this code as long as you include the original license present in this repository in it.

For any question or if you want to contribute to this project, feel free to contact me or open a pull request.

## Improvements to make
- [x] Reformat code in packages
- [ ] Reformat classes to follow design patterns and Python conventions
- [ ] Add argparser to run the script with various settings using the command line
- [ ] Improve perfomances of the script by minimizing image processing steps
- [ ] Improve pose estimation using more/all the Dlib predicted face keypoints
- [ ] Improve robustness of gaze detection

