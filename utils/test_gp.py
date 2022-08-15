import numpy as np
import sys
sys.path.append('/home/hardik/Desktop/Research/Adversary-CBF')
from mvgp import MVGP

import matplotlib 
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

train_x = np.linspace(0,10,100).reshape(1,-1)

train_y1 = np.cos(train_x)
train_y2 = np.sin(train_x)
omega = np.array([ [1.0, 0.0],[0.0, 1.0] ])
sigma = 0.2
l = 2.0


train_y = np.append( train_y1, train_y2, axis=0 )

gp = MVGP( X = train_x.T, Y = train_y.T, omega = omega, sigma = sigma, l = l, noise = 0.05, horizon=300 )
gp.X_obs = gp.X
gp.Y_obs = gp.Y
gp.N_data = gp.X_obs.shape[0]
gp.resample( n_samples = 100 )
gp.resample_obs( n_samples = 100 )
gp.get_obs_covariance()

ys = []
covs = []
for i in range(train_x.shape[1]):
    mu, cov = gp.predict( train_x[:,i].T )
    # print(mu,cov)
    if ys == []:
        ys = np.copy(mu)
        covs = np.diagonal( cov ).reshape(1,-1)
    else:
        ys = np.append( ys, mu, axis=0 )
        covs = np.append( covs, np.diagonal( cov ).reshape(1,-1) , axis=0)
        
## Plot

factor = 5

fig1, axis1 = plt.subplots(2,1)
axis1[0].plot( train_x[0], train_y1[0], 'k--', label='True Value' )
axis1[0].plot( train_x[0], ys[:,0], 'r', label='Untrained Predicted Mean' )
axis1[0].fill_between( train_x[0], ys[:,0] - factor * covs[:,0], ys[:,0] + factor * covs[:,0], color="tab:orange", alpha=0.2 )

axis1[1].plot( train_x[0], train_y2[0], label='True Value' )
axis1[1].plot( train_x[0], ys[:,1], label='Untrained Predicted Mean' )
axis1[1].fill_between( train_x[0], ys[:,1] - factor * covs[:,1], ys[:,1] + factor * covs[:,1], color="tab:orange", alpha=0.2 )

gp.train(max_iters=10)
print(f"New Parameters: omage: {gp.omega}, Sigma: {gp.sigma}, l:{gp.l}")

# gp.omega = np.array( [ [0.9168, -0.0205],[-0.0205, 0.9809] ] )
# gp.sigma = 0.6739
# gp.l = 1.5527

gp.get_obs_covariance()


ys = []
covs = []
for i in range(train_x.shape[1]):
    mu, cov = gp.predict( train_x[:,i].T )
    # print(mu,cov)
    if ys == []:
        ys = np.copy(mu)
        covs = np.diagonal( cov ).reshape(1,-1)
    else:
        ys = np.append( ys, mu, axis=0 )
        covs = np.append( covs, np.diagonal( cov ).reshape(1,-1) , axis=0)
        
## Plot
axis1[0].plot( train_x[0], ys[:,0], 'g', label='Trained Predicted Mean' )
axis1[0].fill_between( train_x[0], ys[:,0] - factor * covs[:,0], ys[:,0] + factor * covs[:,0], color="tab:blue", alpha=0.2 )

axis1[1].plot( train_x[0], ys[:,1], 'g', label='Trained Predicted Mean' )
axis1[1].fill_between( train_x[0], ys[:,1] - factor * covs[:,1], ys[:,1] + factor * covs[:,1], color="tab:blue", alpha=0.2 )

plt.show()