
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
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        # PF needs no Jacobian
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
        #       It is statistically more random.                                      #
        ###############################################################################
        
        u_noise_std = np.linalg.cholesky(self.M(u))
        for j in range(self.n):
            sample_action = u.reshape(-1,1) + u_noise_std @ rng.standard_normal((3,1))
            self.particles[:,j] = self.gfun(self.particles[:,j], sample_action.reshape(3))

    def correction(self, z, landmarks):
        ###############################################################################
        # TODO: Implement the correction step for PF                                  #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        # Hint: landmarks -- (6, 2) landmarks 2d positions in world coordniate        #
        #       z         -- (6, ) each half is [bearing, range, landmark_id]         #
        ###############################################################################
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        
        weight = np.zeros(self.n)
        for i in range(self.n):
            z_hat1 = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], self.particles[:,i])
            z_hat2 = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], self.particles[:,i])
            diff = np.array([
                wrap2Pi(z[0] - z_hat1[0]),
                z[1] - z_hat1[1],
                wrap2Pi(z[3] - z_hat2[0]),
                z[4] - z_hat2[1]])
            weight[i] = multivariate_normal.pdf(diff, np.zeros(4), cov=block_diag(self.Q, self.Q))

        self.particle_weight *= weight
        self.particle_weight /= np.sum(self.particle_weight)
        
        # effective sample size
        Neff = 1 / np.sum(self.particle_weight ** 2)
        if Neff < self.n / 5:
            self.resample()

        # compute the final pose
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
            cosSum += np.cos(self.particles[2,s]) * self.particle_weight[s]
            sinSum += np.sin(self.particles[2,s]) * self.particle_weight[s]
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
