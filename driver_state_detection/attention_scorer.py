from collections import deque


class AttentionScorer:
    """Turn eye, gaze, and pose observations into time-based driver states.

    ``None`` observations are unknown: they neither add to nor decay alert timers.
    Rolling PERCLOS is time-weighted and excludes unknown intervals.
    """

    def __init__(
        self,
        t_now,
        ear_thresh,
        gaze_thresh,
        perclos_thresh=0.2,
        perclos_time_period=60.0,
        perclos_min_valid_fraction=0.8,
        roll_thresh=20,
        pitch_thresh=20,
        yaw_thresh=20,
        ear_time_thresh=2.0,
        gaze_time_thresh=2.0,
        pose_time_thresh=2.5,
        decay_factor=0.9,
        verbose=False,
    ):
        """Configure value thresholds, time thresholds, decay, and PERCLOS coverage."""

        # Thresholds and configuration
        self.ear_thresh = ear_thresh
        self.gaze_thresh = gaze_thresh
        self.perclos_thresh = perclos_thresh
        self.perclos_time_period = perclos_time_period
        self.perclos_min_valid_fraction = perclos_min_valid_fraction
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

        self.perclos_samples = deque()
        self.perclos_ready = False
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
        condition : bool or None
            True accumulates, False decays, and None preserves the timer.
        elapsed : float
            Time elapsed since the last update.

        Returns
        -------
        float
            The updated metric value.
        """
        if condition is None:
            return metric_value
        if condition:
            return metric_value + elapsed
        else:
            return metric_value * self.decay_factor**elapsed

    def mark_face_missing(self, t_now):
        """Advance timers without treating missing observations as driver state."""
        self.last_eval_time = t_now
        self.perclos_samples.append((t_now, None))
        self._trim_perclos_samples(t_now)
        self._calculate_rolling_perclos(t_now)

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
            True if closure time reaches the configured closure-time threshold.
        looking_away : bool
            True if the gaze timer reaches its configured time threshold.
        distracted : bool
            True if the pose timer reaches its configured time threshold.
        """
        elapsed = max(0.0, t_now - self.last_eval_time)
        self.last_eval_time = t_now

        ear_condition = None if ear_score is None else ear_score <= self.ear_thresh
        self.closure_time = self._update_metric(
            self.closure_time, ear_condition, elapsed
        )

        gaze_condition = None if gaze_score is None else gaze_score > self.gaze_thresh
        self.not_look_ahead_time = self._update_metric(
            self.not_look_ahead_time, gaze_condition, elapsed
        )

        if head_roll is None and head_pitch is None and head_yaw is None:
            head_condition = None
        else:
            head_condition = (
                (head_roll is not None and abs(head_roll) > self.roll_thresh)
                or (head_pitch is not None and abs(head_pitch) > self.pitch_thresh)
                or (head_yaw is not None and abs(head_yaw) > self.yaw_thresh)
            )
        self.distracted_time = self._update_metric(
            self.distracted_time, head_condition, elapsed
        )

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

    def get_PERCLOS(self, t_now, fps, ear_score):
        """Compute legacy frame-count PERCLOS over a resetting fixed window.

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

        delta = t_now - self.prev_time
        tired = False

        all_frames_numbers_in_perclos_duration = int(self.perclos_time_period * fps)
        if all_frames_numbers_in_perclos_duration <= 0:
            return False, 0.0

        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.eye_closure_counter += 1

        perclos_score = (
            self.eye_closure_counter
        ) / all_frames_numbers_in_perclos_duration

        if perclos_score >= self.perclos_thresh:
            tired = True

        if delta >= self.perclos_time_period:
            self.eye_closure_counter = 0
            self.prev_time = t_now

        return tired, perclos_score

    def get_rolling_PERCLOS(self, t_now, ear_score):
        """
        Compute time-weighted eye closure over a rolling window.

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
        eye_closed = None if ear_score is None else ear_score <= self.ear_thresh
        self.perclos_samples.append((t_now, eye_closed))
        self._trim_perclos_samples(t_now)

        perclos_score = self._calculate_rolling_perclos(t_now)
        tired = self.perclos_ready and perclos_score >= self.perclos_thresh
        return tired, perclos_score

    def _calculate_rolling_perclos(self, t_now):
        """Integrate known eye states and update rolling-window readiness."""

        closed_duration = 0.0
        valid_duration = 0.0
        samples = list(self.perclos_samples)
        window_start = t_now - self.perclos_time_period
        # Each state applies until the next sample, so integrate sample intervals.
        for (start, closed), (end, _) in zip(samples, samples[1:]):
            interval_start = max(start, window_start)
            duration = max(0.0, end - interval_start)
            if closed is not None:
                valid_duration += duration
                if closed:
                    closed_duration += duration

        perclos_score = closed_duration / valid_duration if valid_duration else 0.0
        window_is_full = bool(samples) and samples[0][0] <= window_start
        enough_valid_data = (
            valid_duration >= self.perclos_time_period * self.perclos_min_valid_fraction
        )
        self.perclos_ready = window_is_full and enough_valid_data
        return perclos_score

    def _trim_perclos_samples(self, t_now):
        """Drop old samples while retaining the state crossing the window boundary."""
        window_start = t_now - self.perclos_time_period
        while (
            len(self.perclos_samples) > 1 and self.perclos_samples[1][0] <= window_start
        ):
            self.perclos_samples.popleft()
