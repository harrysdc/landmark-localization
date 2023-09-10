
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
        self.M = system.M # covariance of motion noise
        self.Q = system.Q # covariance of measurement noise
        
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
        # TODO: Implement the prediction step for PF                                  #
        # Hint: Propagate your particles. Particles are saved in self.particles       #
        # Hint: Use rng.standard_normal instead of np.random.randn.                   #
        #       It is statistically more random.#                                     #
        ###############################################################################
        '''
        u is control input of shape (3,)
        PF needs no Jacobian
        '''
        
        for i in range(self.n):
            noise = self.M(u).diagonal() * rng.standard_normal(3)
            u += noise
            self.particles[:, i] = self.gfun(self.particles[:, i], u)

    def correction(self, z, landmarks):
        ###############################################################################
        # TODO: Implement the correction step for PF                                  #
        # Hint: self.mean_variance() will update the mean and covariance              #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        '''
        landmarks -- (6, 2) landmarks 2d positions in world coordniate
        z -- (6, ) each half is [bearing, range, landmark_id]
        '''
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        
        def likelihood(z, x, landmark1, landmark2):
            '''
            z -- (6, ) each half is [bearing, range, landmark_id]
            x -- robot state (3, )
            '''
            # compute expected bearing and range to landmark1 and landmark2 for a particle
            # compute difference in expected measurement and real measurement
            z1_diff = z[0:2] - self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], x)
            z2_diff = z[3:5] - self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], x)
            z1_diff[1] = wrap2Pi(z1_diff[1])
            z2_diff[1] = wrap2Pi(z2_diff[1])

            # compute the likelihood of the measurements for each particle
            likelihoods1 = multivariate_normal.pdf(z1_diff, mean=np.zeros(2), cov=self.Q)
            likelihoods2 = multivariate_normal.pdf(z2_diff, mean=np.zeros(2), cov=self.Q)
            likelihoods = likelihoods1 * likelihoods2
            return likelihoods

        for i in range(self.n):
            self.particle_weight[i] = likelihood(z, self.particles[:, i], landmark1, landmark2)

        # normalize weights
        self.particle_weight /= np.sum(self.particle_weight)

        # compute posterior pose
        self.mean_variance()

    def resample(self):
        ###############################################################################
        # TODO: Low-variance resampling                                               #
        # Draws a random number r and select those particles that correspond to       #
        # u = r + (m-1) * M^-1 where m = 1,...,M                                      #
        # The sequential stochastic process allows selection with probability         #
        # proportional to the sample weight and has a low complexity of O(M).         #
        ###############################################################################
        
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
        ###############################################################################
        # TODO: Compute the final pose estimate based on the posterior distribution   #
        # Use circular mean for computing average of theta                            #
        ###############################################################################
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
