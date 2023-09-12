
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
        X = self.state_.getState() # (3, )
        P = self.state_.getCovariance() # (3, 3)

        self.kappa_g = 0
        self.sigma_point(X.reshape((3,1)), P, self.kappa_g)

        # propagate sigma points
        for i in range(2 * self.n+1):
            self.X[:, i] = self.gfun(self.X[:, i], u)

        # predict the next state and update covariance
        X_pred = np.zeros_like(self.X)
        X_pred = np.sum(self.w * self.X, axis=1, keepdims=True) # (3, 1)
        X_diff = self.X - X_pred # (3, 7)
        P_pred = (self.w * X_diff) @ X_diff.T + self.M(u)
        
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

        Z = np.zeros((4, 2 * self.n + 1))
        z_hat = np.zeros((4, 1))
        for i in range(Z.shape[1]):
            Z[:2, i] = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], self.X[:, i])
            Z[2:, i] = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], self.X[:, i])
            z_hat += self.w[i] * Z[:,i].reshape(-1,1)
        
        # kalman gain
        P = (Z - z_hat) @ np.diag(self.w) @ (Z - z_hat).T + block_diag(self.Q, self.Q)
        cross_cov = (self.Y - X_predict) @ np.diag(self.w) @ (Z - z_hat).T
        K = cross_cov @ np.linalg.inv(P)

        # correct state and variance
        diff = [
            wrap2Pi(z[0] - z_hat[0]),
            z[1] - z_hat[1],
            wrap2Pi(z[3] - z_hat[2]),
            z[4] - z_hat[3]]
        X = X_predict + K @ diff
        X[2] = wrap2Pi(X[2])
        P = P_predict - K @ P @ K.T

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X.reshape(3))
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
