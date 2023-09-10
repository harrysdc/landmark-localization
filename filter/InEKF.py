
from mimetypes import init
from os import stat
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi
from scipy.linalg import logm, expm


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        self.gfun = system.gfun    # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun    # Jacobian of motion model w.r.t state
        # self.Vfun = init.Vfun    # Jacobian of motion model w.r.t motion noise
        # self.Hfun = init.Hfun    # Jocabian of measurement model
        self.W = system.W          # covariance of motion noise 
        self.V = system.V          # covariance of measurement noise
        
        self.mu = init.mu # (3, 3)
        self.Sigma = init.Sigma # (3, 3)

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])]) # (3,)
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u) # (3,)
        H_pred = self.pose_mat(state_pred)
        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################
        # init 3x3 adjoint function for propagating covariance
        adjX = np.hstack((self.mu[0:2, 0:2], np.array([[self.mu[1, 2]], [-self.mu[0, 2]]])))
        adjX = np.vstack((adjX, np.array([0,0,1])))
        self.propagation(u_se2, adjX)

    def propagation(self, u, adjX):
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################
        self.X_pred = self.mu @ expm(u) # (3, 3)
        self.P_pred =  self.Sigma + adjX @ self.W @ adjX.T # (3, 3)

        
    def correction(self, Y1, Y2, z, landmarks):
        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        lm1_x = landmark1.getPosition()[0]
        lm1_y = landmark1.getPosition()[1]
        lm2_x = landmark2.getPosition()[0]
        lm2_y = landmark2.getPosition()[1]
        H1 = np.array([[lm1_y, -1, 0], [-lm1_x, 0, -1]])
        H2 = np.array([[lm2_y, -1, 0], [-lm2_x, 0, -1]])
        H = np.vstack((H1, H2)) # (4, 3)

        N = self.X_pred @ np.diag([100000,100000,0]) @ self.X_pred.T
        N = N[0:2, 0:2]
        S = H @ self.P_pred @ H.T + block_diag(N, N) # (4,4)
        L = self.P_pred @ H.T @ np.linalg.inv(S) # (3, 4)

        eta1 = self.X_pred @ Y1.reshape((3,1)) - np.array([[lm1_x], [lm1_y], [1]])
        eta2 = self.X_pred @ Y2.reshape((3,1)) - np.array([[lm2_x], [lm2_y], [1]])
        eta = np.squeeze(np.vstack((eta1[0:2], eta2[0:2])))

        twist = L @ eta # (3,)
        twist_hat = self.wedge(twist)

        self.mu = expm(twist_hat) @ self.X_pred
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        self.Sigma = (np.eye(3)-L@H) @ self.P_pred @ (np.eye(3)-L@H).T + L @ block_diag(N, N) @ L.T # (3,3)

        # update RobotState()
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H
    
    def wedge(self, X):
        '''
        wedge operation for se(2) to put an R^3 vector to the Lie algebra basis
        '''
        G1=np.array([[0,-1,0],[1,0,0],[0,0,0]])# omega        
        G2=np.array([[0,0,1],[0,0,0],[0,0,0]]) # v_1
        G3=np.array([[0,0,0],[0,0,1],[0,0,0]]) # v_2
        x_hat = G1 * X[0] + G2 *X[1] + G3 * X[2]
        return x_hat
