import time
import numpy as np


class AttentionScorer:
    """
    Attention Scorer class that contains methods for estimating EAR, Gaze_Score, PERCLOS and Head Pose over time,
    with the given thresholds (time thresholds and value thresholds)

    Methods
    ----------
    - eval_scores: used to evaluate the driver's state of attention
    - get_PERCLOS: specifically used to evaluate the driver sleepiness
    """

    def __init__(
        self,
        t_now,
        ear_thresh,
        gaze_thresh,
        perclos_thresh=0.2,
        roll_thresh=60,
        pitch_thresh=20,
        yaw_thresh=30,
        ear_time_thresh=4.0,
        gaze_time_thresh=2.0,
        pose_time_thresh=4.0,
        decay_factor=0.9,
        verbose=False,
    ):
        """
        Initialize the AttentionScorer object with the given thresholds and parameters.

        Parameters
        ----------
        t_now: float or int
            The current time in seconds.

        ear_thresh: float or int
            EAR score value threshold (if the EAR score is less than this value, eyes are considered closed!)

        gaze_thresh: float or int
            Gaze Score value threshold (if the Gaze Score is more than this value, the gaze is considered not centered)

        perclos_thresh: float (ranges from 0 to 1), optional
            PERCLOS threshold that indicates the maximum time allowed in 60 seconds of eye closure
            (default is 0.2 -> 20% of 1 minute)

        roll_thresh: int, optional
            The roll angle increases or decreases when you turn your head clockwise or counter clockwise.
            Threshold of the roll angle for considering the person distracted/unconscious (not straight neck)
            Default threshold is 20 degrees from the center position.

        pitch_thresh: int, optional
            The pitch angle increases or decreases when you move your head upwards or downwards.
            Threshold of the pitch angle for considering the person distracted (not looking in front)
            Default threshold is 20 degrees from the center position.

        yaw_thresh: int, optional
            The yaw angle increases or decreases when you turn your head to left or right.
            Threshold of the yaw angle for considering the person distracted/unconscious (not straight neck)
            It increase or decrease when you turn your head to left or right. default is 20 degrees from the center position.

        ear_time_thresh: float or int, optional
            Maximum time allowable for consecutive eye closure (given the EAR threshold considered)
            (default is 4.0 seconds)

        gaze_time_thresh: float or int, optional
            Maximum time allowable for consecutive gaze not centered (given the Gaze Score threshold considered)
            (default is 2.0 seconds)

        pose_time_thresh: float or int, optional
            Maximum time allowable for consecutive distracted head pose (given the pitch,yaw and roll thresholds)
            (default is 4.0 seconds)

        decay_factor: float, optional
            Decay factor for the attention scores. This value should be between 0 and 1. The decay factor is used to reduce the score over time when a distraction condition is not met, simulating a decay effect. A value of 0 means istant decay to 0, while a value of 1 means the score will not decay at all. (default is 0.9)

        verbose: bool, optional
            If set to True, print additional information about the scores (default is False)
        """

        # Thresholds and configuration
        self.ear_thresh = ear_thresh
        self.gaze_thresh = gaze_thresh
        self.perclos_thresh = perclos_thresh
        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.ear_time_thresh = ear_time_thresh
        self.gaze_time_thresh = gaze_time_thresh
        self.pose_time_thresh = pose_time_thresh
        self.decay_factor = decay_factor
        self.verbose = verbose

        # Initialize timers for smoothing the metrics
        self.last_eval_time = t_now
        self.closure_time = 0.0
        self.not_look_ahead_time = 0.0
        self.distracted_time = 0.0

        # PERCLOS parameters
        self.PERCLOS_TIME_PERIOD = 60
        self.timestamps = np.empty((0,), dtype=np.float64)
        self.closed_flags = np.empty((0,), dtype=bool)
        self.eye_closure_counter = 0
        self.prev_time = t_now

    def _update_metric(self, metric_value, condition, elapsed):
        """
        Update a given metric timer based on the condition.

        If the condition is True, accumulate the elapsed time.
        Otherwise, apply exponential decay to the metric value.

        Parameters
        ----------
        metric_value : float
            The current accumulated value of the metric.
        condition : bool
            True if the current measurement should accumulate more time.
        elapsed : float
            Time elapsed since the last update.

        Returns
        -------
        float
            The updated metric value.
        """
        if condition:
            return metric_value + elapsed
        else:
            return metric_value * self.decay_factor

    def eval_scores(
        self, t_now, ear_score, gaze_score, head_roll, head_pitch, head_yaw
    ):
        """
        Evaluate the driver's state of attention using smoothed metrics.

        Instead of instantly resetting timers when conditions are not met,
        each timer is updated with accumulated elapsed time when active or decayed otherwise.

        Parameters
        ----------
        t_now : float or int
            The current time in seconds.
        ear_score : float
            The Eye Aspect Ratio (EAR) score.
        gaze_score : float
            The gaze score.
        head_roll : float
            The roll angle of the head.
        head_pitch : float
            The pitch angle of the head.
        head_yaw : float
            The yaw angle of the head.

        Returns
        -------
        asleep : bool
            True if the accumulated closure time exceeds the EAR threshold.
        looking_away : bool
            True if the accumulated gaze timer exceeds its threshold.
        distracted : bool
            True if the accumulated head pose timer exceeds its threshold.
        """
        # Calculate the time elapsed since the last evaluation
        elapsed = t_now - self.last_eval_time
        self.last_eval_time = t_now

        # Update the eye closure metric
        self.closure_time = self._update_metric(
            self.closure_time,
            (ear_score is not None and ear_score <= self.ear_thresh),
            elapsed,
        )

        # Update the gaze metric
        self.not_look_ahead_time = self._update_metric(
            self.not_look_ahead_time,
            (gaze_score is not None and gaze_score > self.gaze_thresh),
            elapsed,
        )

        # Update the head pose metric: check if any head angle exceeds its threshold
        head_condition = (
            (head_roll is not None and abs(head_roll) > self.roll_thresh)
            or (head_pitch is not None and abs(head_pitch) > self.pitch_thresh)
            or (head_yaw is not None and abs(head_yaw) > self.yaw_thresh)
        )
        self.distracted_time = self._update_metric(
            self.distracted_time, head_condition, elapsed
        )

        # Determine driver state based on thresholds
        asleep = self.closure_time >= self.ear_time_thresh
        looking_away = self.not_look_ahead_time >= self.gaze_time_thresh
        distracted = self.distracted_time >= self.pose_time_thresh

        if self.verbose:
            print(
                f"Closure Time: {self.closure_time:.2f}s | "
                f"Not Look Ahead Time: {self.not_look_ahead_time:.2f}s | "
                f"Distracted Time: {self.distracted_time:.2f}s"
            )

        return asleep, looking_away, distracted

    # NOTE: This method uses a fixed window for the PERCLOS score - that is it resets every X seconds and don't consider the last X seconds as a rolling window!
    def get_PERCLOS(self, t_now, fps, ear_score):
        """
        Compute the PERCLOS (Percentage of Eye Closure) score over a given time period.

        Parameters
        ----------
        t_now: float or int
            The current time in seconds.

        fps: int
            The frames per second of the video.

        ear_score: float
            EAR (Eye Aspect Ratio) score obtained from the driver eye aperture.

        Returns
        -------
        tired: bool
            Indicates if the driver is tired or not.

        perclos_score: float
            The PERCLOS score over a minute.
        """

        delta = t_now - self.prev_time  # set delta timer
        tired = False  # set default value for the tired state of the driver

        all_frames_numbers_in_perclos_duration = int(self.PERCLOS_TIME_PERIOD * fps)

        # if the ear_score is lower or equal than the threshold, increase the eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.eye_closure_counter += 1

        # compute the PERCLOS over a given time period
        perclos_score = (
            self.eye_closure_counter
        ) / all_frames_numbers_in_perclos_duration

        if (
            perclos_score >= self.perclos_thresh
        ):  # if the PERCLOS score is higher than a threshold, tired = True
            tired = True

        if (
            delta >= self.PERCLOS_TIME_PERIOD
        ):  # at every end of the given time period, reset the counter and the timer
            self.eye_closure_counter = 0
            self.prev_time = t_now

        return tired, perclos_score

    def get_rolling_PERCLOS(self, t_now, ear_score):
        """
        Compute the rolling PERCLOS score using NumPy vectorized operations.

        Parameters
        ----------
        t_now : float or int
            The current time in seconds.
        ear_score : float
            The EAR (Eye Aspect Ratio) score for the current frame.

        Returns
        -------
        tired : bool
            Indicates if the driver is tired based on the PERCLOS score.
        perclos_score : float
            The rolling PERCLOS score calculated over the defined time period.
        """
        # Determine if the current frame indicates closed eyes
        eye_closed = (ear_score is not None) and (ear_score <= self.ear_thresh)

        # Append new values to the NumPy arrays. (np.concatenate creates new arrays.)
        self.timestamps = np.concatenate((self.timestamps, [t_now]))
        self.closed_flags = np.concatenate((self.closed_flags, [eye_closed]))

        # Create a boolean mask of entries within the rolling window.
        valid_mask = self.timestamps >= (t_now - self.PERCLOS_TIME_PERIOD)
        self.timestamps = self.timestamps[valid_mask]
        self.closed_flags = self.closed_flags[valid_mask]

        total_frames = self.timestamps.size
        if total_frames > 0:
            perclos_score = np.sum(self.closed_flags) / total_frames
        else:
            perclos_score = 0.0

        tired = perclos_score >= self.perclos_thresh
        return tired, perclos_score
