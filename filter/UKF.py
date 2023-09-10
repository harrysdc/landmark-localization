
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi, unscented_propagate


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # covariance of motion noise 
        self.Q = system.Q # covariance of measurement noise
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    def prediction(self, u):
        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        # prior belief
        X = self.state_.getState() # (3, )
        P = self.state_.getCovariance() # (3, 3)

        self.kappa_g = 0
        self.sigma_point(X.reshape((3,1)), P, self.kappa_g)
        
        for i in range(2 * self.n+1):
            self.X[:, i] = self.gfun(self.X[:, i], u)
        X_pred = np.zeros_like(self.X)
        X_pred = np.sum(self.w * self.X, axis=1, keepdims=True) # (3, 1)

        X_diff = self.X - X_pred # (3, 7)
        P_pred = (self.w * X_diff) @ X_diff.T + self.M(u)
        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)

    def correction(self, z, landmarks):
        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        def correct(z, lm, X_predict, P_predict):
            self.sigma_point(X_predict.reshape((3,1)), P_predict, self.kappa_g)
            
            z_expected = np.zeros((2,7))
            for i in range(2*self.n+1): 
                z_expected[:, i] = self.hfun(lm.getPosition()[0], lm.getPosition()[1], self.X[:, i])
                z_expected[1, i] = wrap2Pi(z_expected[1, i])

            z_mean = np.sum(self.w * z_expected, axis=1, keepdims=True)
            z_mean[1] = wrap2Pi(z_mean[1])
            z_diff = z_expected - z_mean
            cov_z = (self.w * z_diff) @ z_diff.T + self.Q # (2, 2)

            X_diff = self.X - X_predict
            crossCov = (self.w * X_diff) @ z_diff.T # (3, 2)
            K = crossCov @ np.linalg.inv(cov_z) # (3, 2)

            X = X_predict + K @ (z.reshape((2,1)) - z_mean)
            P = P_predict - K @ cov_z @ K.T
            X = X.squeeze()
            return X, P

        X_predict, P_predict = correct(z[0:2], landmark1, X_predict, P_predict)
        X, P = correct(z[3:5], landmark2, X_predict.reshape((3,1)), P_predict)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)

    def sigma_point(self, mean, cov, kappa):
        ###############################################################################
        # TODO: calculate sigma points using Cholesky decomposition                   #
        ###############################################################################
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(self.n, axis=1)
        self.X = np.hstack((mean, Y+L, Y-L)) # (3, 7)
        self.w = np.zeros([2 * self.n + 1, 1]) # (7, )
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
