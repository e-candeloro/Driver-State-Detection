import time


class AttentionScorer:

    def __init__(self, capture_fps: int, ear_thresh, gaze_thresh, perclos_thresh=0.2, ear_time_thresh=4.0, pitch_thresh=20,
                 yaw_thresh=30, gaze_time_thresh=4.0, roll_thresh=60, pose_time_thresh=4.0, verbose=False):
        """
        Attention Scorer class that contains methods for estimating EAR, Gaze_Score, PERCLOS and Head Pose over time,
        with the given thresholds (time thresholds and value thresholds)

        Parameters
        ----------
        capture_fps: int
            Upper frame rate of video/capture stream considered

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

        self.fps = capture_fps
        self.delta_time_frame = (1.0 / capture_fps)  # estimated frame time
        self.prev_time = 0  # auxiliary variable for the PERCLOS estimation function
        # default time period for PERCLOS (60 seconds)
        self.perclos_time_period = 60
        self.perclos_thresh = perclos_thresh

        # the time thresholds are divided for the estimated frame time
        # (that is a function passed parameter and so can vary)
        self.ear_thresh = ear_thresh
        self.ear_act_thresh = ear_time_thresh / self.delta_time_frame
        self.ear_counter = 0
        self.eye_closure_counter = 0

        self.gaze_thresh = gaze_thresh
        self.gaze_act_thresh = gaze_time_thresh / self.delta_time_frame
        self.gaze_counter = 0

        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.pose_act_thresh = pose_time_thresh / self.delta_time_frame
        self.pose_counter = 0

        self.verbose = verbose

    def eval_scores(self, ear_score, gaze_score, head_roll, head_pitch, head_yaw):
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

        if self.ear_counter >= self.ear_act_thresh:  # check if the ear cumulative counter surpassed the threshold
            asleep = True

        if self.gaze_counter >= self.gaze_act_thresh:  # check if the gaze cumulative counter surpassed the threshold
            looking_away = True

        if self.pose_counter >= self.pose_act_thresh:  # check if the pose cumulative counter surpassed the threshold
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
            if not asleep:
                self.ear_counter += 1
        elif self.ear_counter > 0:
            self.ear_counter -= 1

        if (gaze_score is not None) and (gaze_score >= self.gaze_thresh):
            if not looking_away:
                self.gaze_counter += 1
        elif self.gaze_counter > 0:
            self.gaze_counter -= 1

        if ((self.roll_thresh is not None and head_roll is not None and abs(head_roll) > self.roll_thresh) or (
                head_pitch is not None and abs(head_pitch) > self.pitch_thresh) or (
                head_yaw is not None and abs(head_yaw) > self.yaw_thresh)):
            if not distracted:
                self.pose_counter += 1
        elif self.pose_counter > 0:
            self.pose_counter -= 1

        if self.verbose:  # print additional info if verbose is True
            print(
                f"ear counter:{self.ear_counter}/{self.ear_act_thresh}\ngaze counter:{self.gaze_counter}/{self.gaze_act_thresh}\npose counter:{self.pose_counter}/{self.pose_act_thresh}")
            print(
                f"eye closed:{asleep}\tlooking away:{looking_away}\tdistracted:{distracted}")

        return asleep, looking_away, distracted

    def get_PERCLOS(self, ear_score):
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

        delta = time.time() - self.prev_time  # set delta timer
        tired = False  # set default value for the tired state of the driver

        # if the ear_score is lower or equal than the threshold, increase the eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.eye_closure_counter += 1

        # compute the cumulative eye closure time
        closure_time = (self.eye_closure_counter * self.delta_time_frame)
        # compute the PERCLOS over a given time period
        perclos_score = (closure_time) / self.perclos_time_period

        if perclos_score >= self.perclos_thresh:  # if the PERCLOS score is higher than a threshold, tired = True
            tired = True

        if self.verbose:
            print(
                f"Closure Time:{closure_time}/{self.perclos_time_period}\nPERCLOS: {round(perclos_score, 3)}")

        if delta >= self.perclos_time_period:  # at every end of the given time period, reset the counter and the timer
            self.eye_closure_counter = 0
            self.prev_time = time.time()

        return tired, perclos_score
