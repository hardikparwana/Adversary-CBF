import numpy as np
import torch
import sys
sys.path.append('/home/hardik/Desktop/Research/Adversary-CBF')
from mvgp import MVGP


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


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

L = 5.0
p = 1.0

gp = MVGP( X = train_x.T, Y = train_y.T, omega = omega, sigma = sigma, L = L, p = p, l = l, noise = 0.05, horizon=300, kernel_type = 'Gaussian' )
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

axis1[1].plot( train_x[0], train_y2[0], 'k--', label='True Value' )
axis1[1].plot( train_x[0], ys[:,1], 'r', label='Untrained Predicted Mean' )
axis1[1].fill_between( train_x[0], ys[:,1] - factor * covs[:,1], ys[:,1] + factor * covs[:,1], color="tab:orange", alpha=0.2 )

gp.train(max_iters=100, print_status = True)
print(f"New Parameters: omega: {gp.omega}, Sigma: {gp.sigma}, l:{gp.l}, L:{gp.L}, p:{gp.p}")
gp.get_obs_covariance()

# gp.omega = np.array( [ [0.9168, -0.0205],[-0.0205, 0.9809] ] )
# gp.sigma = 0.6739
# gp.l = 1.5527

# ys = []
# covs = []
# for i in range(train_x.shape[1]):
#     mu, cov = gp.predict( train_x[:,i].T )
#     # print(mu,cov)
#     if ys == []:
#         ys = np.copy(mu)
#         covs = np.diagonal( cov ).reshape(1,-1)
#     else:
#         ys = np.append( ys, mu, axis=0 )
#         covs = np.append( covs, np.diagonal( cov ).reshape(1,-1) , axis=0)
        
# ## Plot
# axis1[0].plot( train_x[0], ys[:,0], 'g', label='Trained Predicted Mean' )
# axis1[0].fill_between( train_x[0], ys[:,0] - factor * covs[:,0], ys[:,0] + factor * covs[:,0], color="tab:blue", alpha=0.2 )

# axis1[1].plot( train_x[0], ys[:,1], 'g', label='Trained Predicted Mean' )
# axis1[1].fill_between( train_x[0], ys[:,1] - factor * covs[:,1], ys[:,1] + factor * covs[:,1], color="tab:blue", alpha=0.2 )

# Pytorch tensor prediction test

gp.initialize_torch()
train_x_tensor = torch.tensor(train_x)

ys = []
covs = []
for i in range(train_x_tensor.shape[1]):
    mu, cov = gp.predict_torch( train_x_tensor[:,i].T )
    # print(mu,cov)
    if ys == []:
        ys = np.copy(mu.detach().numpy())
        covs = np.diagonal( cov.detach().numpy() ).reshape(1,-1)
    else:
        ys = np.append( ys, mu.detach().numpy(), axis=0 )
        covs = np.append( covs, np.diagonal( cov.detach().numpy() ).reshape(1,-1) , axis=0)
        
## Plot
axis1[0].plot( train_x[0], ys[:,0], 'g', label='Trained Predicted Mean' )
axis1[0].fill_between( train_x[0], ys[:,0] - factor * covs[:,0], ys[:,0] + factor * covs[:,0], color="tab:blue", alpha=0.2 )
axis1[0].legend()

axis1[1].plot( train_x[0], ys[:,1], 'g', label='Trained Predicted Mean' )
axis1[1].fill_between( train_x[0], ys[:,1] - factor * covs[:,1], ys[:,1] + factor * covs[:,1], color="tab:blue", alpha=0.2 )
axis1[1].legend()

plt.show()