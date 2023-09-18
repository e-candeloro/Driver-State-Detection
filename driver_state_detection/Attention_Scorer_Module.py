import time


class AttentionScorer:

    def __init__(self, t_now, ear_thresh, gaze_thresh, perclos_thresh=0.2, roll_thresh=60,
                 pitch_thresh=20, yaw_thresh=30, ear_time_thresh=4.0, gaze_time_thresh=2.,
                 pose_time_thresh=4.0, verbose=False):
        """
        Attention Scorer class that contains methods for estimating EAR, Gaze_Score, PERCLOS and Head Pose over time,
        with the given thresholds (time thresholds and value thresholds)

        Parameters
        ----------
        ear_thresh: float or int
            EAR score value threshold (if the EAR score is less than this value, eyes are considered closed!)

        gaze_thresh: float or int
            Gaze Score value threshold (if the Gaze Score is more than this value, the gaze is considered not centered)

        perclos_thresh: float (ranges from 0 to 1)
            PERCLOS threshold that indicates the maximum time allowed in 60 seconds of eye closure
            (default is 0.2 -> 20% of 1 minute)

        ear_time_thresh: float or int
            Maximum time allowable for consecutive eye closure (given the EAR threshold considered)
            (default is 4.0 seconds)

            
        roll_thresh: int
            The roll angle increases or decreases when you turn your head clockwise or counter clockwise.
            Threshold of the roll angle for considering the person distracted/unconscious (not straight neck)
            Default threshold is 20 degrees from the center position.
        
        pitch_thresh: int
            The pitch angle increases or decreases when you move your head upwards or downwards.
            Threshold of the pitch angle for considering the person distracted (not looking in front)
            Default threshold is 20 degrees from the center position.

        yaw_thresh: int
            The yaw angle increases or decreases when you turn your head to left or right.
            Threshold of the yaw angle for considering the person distracted/unconscious (not straight neck)
            It increase or decrease when you turn your head to left or right. default is 20 degrees from the center position.

        pose_time_thresh: float or int
            Maximum time allowable for consecutive distracted head pose (given the pitch,yaw and roll thresholds)
            (default is 4.0 seconds)

        verbose: bool
            If set to True, print additional information about the scores (default is False)


        Methods
        ----------
        - eval_scores: used to evaluate the driver's state of attention
        - get_PERCLOS: specifically used to evaluate the driver sleepiness
        """

        self.ear_thresh = ear_thresh
        self.gaze_thresh = gaze_thresh
        self.perclos_thresh = perclos_thresh
        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.ear_time_thresh = ear_time_thresh
        self.gaze_time_thresh = gaze_time_thresh
        self.pose_time_thresh = pose_time_thresh
        self.verbose = verbose

        self.perclos_time_period = 60
        
        self.last_time_eye_opened = t_now
        self.last_time_looked_ahead = t_now
        self.last_time_attended = t_now
        self.closure_time = 0
        self.not_look_ahead_time = 0
        self.distracted_time = 0

        self.prev_time = t_now
        self.eye_closure_counter = 0


    def eval_scores(self, t_now, ear_score, gaze_score, head_roll, head_pitch, head_yaw):
        """
        :param ear_score: float
            EAR (Eye Aspect Ratio) score obtained from the driver eye aperture
        :param gaze_score: float
            Gaze Score obtained from the driver eye gaze
        :param head_roll: float
            Roll angle obtained from the driver head pose
        :param head_pitch: float
            Pitch angle obtained from the driver head pose
        :param head_yaw: float
            Yaw angle obtained from the driver head pose

        :return:
            Returns a tuple of boolean values that indicates the driver state of attention
            tuple: (asleep, looking_away, distracted)
        """
        # instantiating state of attention variables
        asleep = False
        looking_away = False
        distracted = False

        if self.closure_time >= self.ear_time_thresh:  # check if the ear cumulative counter surpassed the threshold
            asleep = True

        if self.not_look_ahead_time >= self.gaze_time_thresh:  # check if the gaze cumulative counter surpassed the threshold
            looking_away = True

        if self.distracted_time >= self.pose_time_thresh:  # check if the pose cumulative counter surpassed the threshold
            distracted = True

        '''
        The 3 if blocks that follow are written in a way that when we have a score that's over it's value threshold, 
        a respective score counter (ear counter, gaze counter, pose counter) is increased and can reach a given maximum 
        over time.
        When a score doesn't surpass a threshold, it is diminished and can go to a minimum of zero.
        
        Example:
        
        If the ear score of the eye of the driver surpasses the threshold for a SINGLE frame, the ear_counter is increased.
        If the ear score of the eye is surpassed for multiple frames, the ear_counter will be increased and will reach 
        a given maximum, then it won't increase but the "asleep" variable will be set to True.
        When the ear_score doesn't surpass the threshold, the ear_counter is decreased. If there are multiple frame
        where the score doesn't surpass the threshold, the ear_counter can reach the minimum of zero
        
        This way, we have a cumulative score for each of the controlled features (EAR, GAZE and HEAD POSE).
        If high score it's reached for a cumulative counter, this function will retain its value and will need a
        bit of "cool-down time" to reach zero again 
        '''
        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.closure_time = t_now - self.last_time_eye_opened
        elif ear_score is None or (ear_score is not None and ear_score > self.ear_thresh):
            self.last_time_eye_opened = t_now
            self.closure_time = 0.

        if (gaze_score is not None) and (gaze_score > self.gaze_thresh):
            self.not_look_ahead_time = t_now - self.last_time_looked_ahead
        elif gaze_score is None or (gaze_score is not None and gaze_score <= self.gaze_thresh):
            self.last_time_looked_ahead = t_now
            self.not_look_ahead_time = 0.

        if ((head_roll is not None and abs(head_roll) > self.roll_thresh) or (
                head_pitch is not None and abs(head_pitch) > self.pitch_thresh) or (
                head_yaw is not None and abs(head_yaw) > self.yaw_thresh)):
            self.distracted_time = t_now - self.last_time_attended
        elif head_roll is None or head_pitch is None or head_yaw is None or (
            (abs(head_roll) <= self.roll_thresh) and (abs(head_pitch) <= self.pitch_thresh) and (
                abs(head_yaw) <= self.yaw_thresh)):
            self.last_time_attended = t_now
            self.distracted_time = 0.

        if self.verbose:  # print additional info if verbose is True
            print(
                f"ear counter:{self.ear_counter}/{self.ear_act_thresh}\ngaze counter:{self.gaze_counter}/{self.gaze_act_thresh}\npose counter:{self.pose_counter}/{self.pose_act_thresh}")
            print(
                f"eye closed:{asleep}\tlooking away:{looking_away}\tdistracted:{distracted}")

        return asleep, looking_away, distracted

    def get_PERCLOS(self, t_now, fps, ear_score):
        """

        :param ear_score: float
            EAR (Eye Aspect Ratio) score obtained from the driver eye aperture
        :return:
            tuple:(tired, perclos_score)

            tired:
                is a boolean value indicating if the driver is tired or not
            perclos_score:
                is a float value indicating the PERCLOS score over a minute
                after a minute this scores resets itself to zero
        """

        delta = t_now - self.prev_time  # set delta timer
        tired = False  # set default value for the tired state of the driver

        all_frames_numbers_in_perclos_duration = int(self.perclos_time_period * fps)

        # if the ear_score is lower or equal than the threshold, increase the eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.eye_closure_counter += 1

        # compute the PERCLOS over a given time period
        perclos_score = (self.eye_closure_counter) / all_frames_numbers_in_perclos_duration

        if perclos_score >= self.perclos_thresh:  # if the PERCLOS score is higher than a threshold, tired = True
            tired = True

        if delta >= self.perclos_time_period:  # at every end of the given time period, reset the counter and the timer
            self.eye_closure_counter = 0
            self.prev_time = t_now

        return tired, perclos_score
