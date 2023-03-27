
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

from scipy.stats import multivariate_normal
from numpy.random import default_rng
rng = default_rng()

class PF:
    # PF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        np.random.seed(2)
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        # PF parameters
        self.n = init.n
        self.Sigma = init.Sigma # (3, 3) diagnoal matrix
        self.particles = init.particles
        self.particle_weight = init.particle_weight # (self.n,)

        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        ###############################################################################
        # TODO: Implement the prediction step for PF, remove pass                     #
        # Hint: Propagate your particles. Particles are saved in self.particles       #
        # Hint: Use rng.standard_normal instead of np.random.randn.                   #
        #       It is statistically more random.                                      #
        ###############################################################################
        '''
        u is control input of shape (3,)
        PF needs no jacobian
        '''
        for i in range(self.n):
            u = u + rng.standard_normal(3)
            self.particles[:, i] = self.gfun(self.particles[:, i], u)
        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################


    def correction(self, z, landmarks):
        '''
        landmarks -- (6, 2) landmarks 2d positions in world coordniate
        z -- (6, ) each half is [bearing, range, landmark_id]
        '''
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        
        ###############################################################################
        # TODO: Implement the correction step for PF                                  #
        # Hint: self.mean_variance() will update the mean and covariance              #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        def likelihood(self, z, x, landmark1, landmark2):
            '''
            z -- (6, ) each half is [bearing, range, landmark_id]
            x -- robot state (3, )
            '''
            bearing1, range1, _ = z[:3]
            bearing2, range2, _ = z[3:]

            lm1_x = landmark1.getPosition()[0]
            lm1_y = landmark1.getPosition()[1]
            lm2_x = landmark2.getPosition()[0]
            lm2_y = landmark2.getPosition()[1]

            par_x = x[0]
            par_y = x[1]
            par_t = x[2]

            # compute expected bearing and range to landmark1 and landmark2 for a particle
            expected_bearing1 = np.arctan2(lm1_y-par_y, lm1_x-par_x) - par_t
            expected_range1 = np.sqrt((lm1_x-par_x)**2 + (lm1_y-par_y)**2)
            expected_bearing2 = np.arctan2(lm2_y - par_y, lm2_x - par_x) - par_t
            expected_range2 = np.sqrt((lm2_x-par_x)**2 + (lm2_y-par_y)**2)

            # compute the likelihood of the measurements for each particle
            likelihoods1 = multivariate_normal.pdf([bearing1, range1], mean=[expected_bearing1, expected_range1], cov=self.Q)
            likelihoods2 = multivariate_normal.pdf([bearing2, range2], mean=[expected_bearing2, expected_range2], cov=self.Q)
            likelihoods = likelihoods1 * likelihoods2
            return likelihoods


        for i in range(self.n):
            self.particle_weight[i] = likelihood(z, self.particles[:, i], landmark1, landmark2)
        
        self.particle_weight /= np.sum(self.particle_weight) # normalize the weights

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.mean_variance()


    def resample(self):
        new_samples = np.zeros_like(self.particles)
        new_weight = np.zeros_like(self.particle_weight)
        W = np.cumsum(self.particle_weight)
        r = np.random.rand(1) / self.n
        count = 0
        for j in range(self.n):
            u = r + j/self.n
            while u > W[count]:
                count += 1
            new_samples[:,j] = self.particles[:,count]
            new_weight[j] = 1 / self.n
        self.particles = new_samples
        self.particle_weight = new_weight
    

    def mean_variance(self):
        X = np.mean(self.particles, axis=1)
        sinSum = 0
        cosSum = 0
        for s in range(self.n):
            cosSum += np.cos(self.particles[2,s])
            sinSum += np.sin(self.particles[2,s])
        X[2] = np.arctan2(sinSum, cosSum)

        zero_mean = np.zeros_like(self.particles)
        for s in range(self.n):
            zero_mean[:,s] = self.particles[:,s] - X
            zero_mean[2,s] = wrap2Pi(zero_mean[2,s])
        P = zero_mean @ zero_mean.T / self.n
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)
    
    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

