import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:
    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jacobian of motion model w.r.t state
        self.Vfun = init.Vfun  # Jacobian of motion model w.r.t motion noise
        self.Hfun = init.Hfun  # Jacobian of measurement model w.r.t state
        self.M = system.M # covariance of motion noise 
        self.Q = system.Q # covariance of measurement noise

        self.state_ = RobotState()

        # initialize state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    def prediction(self, u):
        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################

        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        X_pred = self.gfun(X, u)
        G = self.Gfun(X, u) # (3, 3)
        V = self.Vfun(X, u)# (3, 3)
        P_pred = G @ P @ G.T + V @ self.M(u) @ V.T

        # Set state in RobotState()
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):
        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: z is measurement. Save your corrected state and cov as X and P        #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        
        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance() # (3,3)
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        def correct(z, landmark, X_predict, P_predict):
            landmark_x = landmark.getPosition()[0]
            landmark_y = landmark.getPosition()[1]
            
            z_expected = self.hfun(landmark_x, landmark_y, X_predict)
            z_diff = z - z_expected
            z_diff[1] = wrap2Pi(z_diff[1])

            H = self.Hfun(landmark_x, landmark_y, X_predict, z_expected)
            K = P_predict @ H.T @ np.linalg.pinv(H @ P_predict @ H.T + self.Q)
            
            X = X_predict + K @ z_diff
            P = (np.eye(3)- K @ H) @ P_predict
            return X, P
        
        X_predict, P_predict = correct(z[0:2], landmark1, X_predict, P_predict)
        X, P = correct(z[3:5], landmark2, X_predict, P_predict)

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
