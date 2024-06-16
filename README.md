# Real Time Driver State Detection
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) 

Real time, webcam based, driver attention state detection and monitoring using Python with the OpenCV and mediapipe libraries.

![driver state detection demo](https://user-images.githubusercontent.com/67196406/173455413-ba95db40-6be5-4d64-9a1d-6c998854130e.gif)

**Note**:
This work is partially based on [this paper](https://www.researchgate.net/publication/327942674_Vision-Based_Driver%27s_Attention_Monitoring_System_for_Smart_Vehicles) for the scores and methods used.

## Mediapipe Update
Thanks to the awesome contribution of [MustafaLotfi](https://github.com/MustafaLotfi), now the script uses the better performing and accurate face keypoints detection model from the [Google Mediapipe library](https://github.com/google/mediapipe).

### Features added:

- 478 face keypoints detection
- Direct iris keypoint detection for gaze score estimation
- Improved head pose estimation using the dynamical canonical face model
- Fixed euler angles function and wrong returned values
- Using time variables to make the code more modular and machine agnostic

**NOTE**: the old mediapipe version can still be found in the "dlib-based" repository branch.

## How Does It Work?

This script searches for the driver face, then use the mediapipe library to predict 478 face and iris keypoints.
The enumeration and location of all the face keypoints/landmarks can be seen [here](https://github.com/e-candeloro/Driver-State-Detection/blob/master/docs/5Mohl.jpg).

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

**MEDIAPIPE DEMO COMING SOON**

**OLD DEMO:**

<video src="https://user-images.githubusercontent.com/67196406/121312501-bb571d00-c905-11eb-8d25-1cd28efc9110.mp4" controls="controls" style="max-width: 100%; height: auto;">
    Your browser does not support the video tag.
</video>


## The Scores Explained

### EAR

**Eye Aspect Ratio** is a normalized score that is useful to understand the rate of aperture of the eyes.
Using the mediapipe face mesh keypoints for each eye (six for each), the eye lenght and width are estimated and using this data the EAR score is computed as explained in the image below:
![EAR](https://user-images.githubusercontent.com/67196406/121489162-18210900-c9d4-11eb-9d2e-765f5ac42286.png)

**NOTE:** the average of the two eyes EAR score is computed

### Gaze Score Estimation

The gaze score gives information about how much the driver is looking away without turning his head.

To understand this, the distance between the eye center and the position of the pupil is computed. The result is then normalized by the eye width that can be different depending on the driver physionomy and distance from the camera.

The below image explains graphically how the Gaze Score for a single eye is computed:
![Gaze Score](https://user-images.githubusercontent.com/67196406/121489746-ab5a3e80-c9d4-11eb-8f33-d34afd0947b4.png)
**NOTE:** the average of the two eyes Gaze Score is computed

### Head Pose Estimation

For the head pose estimation, a standard 3d head model in world coordinates was considered, in combination of the respective face mesh keypoints in the image plane. 
In this way, using the solvePnP function of OpenCV, estimating the rotation and translation vector of the head in respect to the camera is possible.
Then the 3 Euler angles are computed.

The partial snippets of code used for this task can be found in [this article](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/).

## Installation

This projects runs on Python with the following libraries:

- numpy
- OpenCV (opencv-python)
- mediapipe

Or you can use poetry to automatically create a virtualenv with all the required packages:

```
pip install poetry #a global install of poetry is required
```

Then inside the repo directory:

```
poetry install
```

To activate the env to execute command lines:

```
poetry shell
```

Alternatively (not recommended), you can use the requirements.txt file provided in the repository using:
    
    pip install -r requirements.txt
    

## Usage

First navigate inside the driver state detection folder:
    
    cd driver_state_detection

The scripts can be used with all default options and parameters by calling it via command line:

    python main.py

For the list of possible arguments, write:

    python main.py --help

Example of a possible use with parameters:

    python main.py --ear_time_tresh 5

This will sets to 5 seconds the eye closure time before a warning  message is shown on screen

## Why this project

This project was developed as part for a final group project for the course of [Computer Vision and Cognitive Systems](https://international.unimore.it/singleins.html?ID=295) done at the [University of Modena and Reggio Emilia](https://international.unimore.it/) in the second semester of the academic year 2020/2021.
Given the possible applications of Computer Vision, we wanted to focus mainly on the automotive field, developing a useful and potential life saving proof of concept project.
In fact, sadly, many fatal accidents happens [because of the driver distraction](https://www.nhtsa.gov/risky-driving/distracted-driving).

## License and Contacts

This project is freely available under the MIT license. You can use/modify this code as long as you include the original license present in this repository in it.

For any question or if you want to contribute to this project, feel free to contact me or open a pull request.

## Improvements to make

- [x] Reformat code in packages
- [x] Add argparser to run the script with various settings using the command line
- [x] Improve robustness of gaze detection (using mediapipe)
- [x] Add argparser option for importing and using the camera matrix and dist. coefficients
- [x] Reformat classes to follow design patterns and Python conventions
- [ ] Debug new mediapipe methods and classes and adjust thresholds
- [ ] Improve perfomances of the script by minimizing image processing steps
