import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

import torch
torch.autograd.set_detect_anomaly(True)

from utils.utils import *
from ut_utils.ut_utilsJIT import *
from utils.mvgp_jit import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from robot_models.custom_cartpole import CustomCartPoleEnv
from gym_wrappers.record_video import RecordVideo
from cartpole_policy import policy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 

def initialize_tensors(robot, param_w, param_mu, param_Sigma):
    x, x_dot, theta, theta_dot = robot.state
    robot.X_torch = torch.tensor( np.array([ x, x_dot, theta, theta_dot ]).reshape(-1,1), requires_grad = True, dtype=torch.float )
    robot.w_torch = torch.tensor( param_w, requires_grad = True, dtype=torch.float )
    robot.mu_torch = torch.tensor( param_mu, requires_grad = True, dtype=torch.float )
    robot.Sigma_torch = torch.tensor( param_Sigma, requires_grad = True, dtype=torch.float )
    
traced_get_mean_JIT = []
traced_policy = []
traced_sigma_point_expand_JIT = []
traced_sigma_point_compress_JIT = []
traced_reward_UT_Mean_Evaluator_basic = []

def get_future_reward(robot, gp_params, K_invs, X_s, Y_s, noise, gps):
           
    prior_states, prior_weights = initialize_sigma_points_JIT(robot.X_torch)
    states = [prior_states]
    weights = [prior_weights]
    reward = torch.tensor([0],dtype=torch.float)
    
    global first_run, traced_get_mean_JIT, traced_policy, traced_sigma_point_expand_JIT, traced_sigma_point_compress_JIT, traced_reward_UT_Mean_Evaluator_basic

    start_t = 1
    if (first_run):
        print("**************************** t ******************************* ", t)
        i = 0
        print("Starting tracing")
        # Get mean position
        traced_get_mean_JIT = get_mean_JIT #torch.jit.trace( get_mean_JIT, (states[i], weights[i] ) )
        mean_position = traced_get_mean_JIT( states[i], weights[i] )
        
        # Get control input
        traced_policy = torch.jit.trace( policy, ( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position ) )        
        solution = traced_policy( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position )
   
        # Get expanded next state
        traced_sigma_point_expand_JIT = sigma_point_expand_JIT #torch.trace.jit( sigma_point_expand_JIT, (torch.tensor(GA, dtype=torch.float), torch.tensor(PE, dtype=torch.float), gp_params, K_invs, noise, X_s, Y_s, states[i], weights[i], solution, dt_outer, dt_inner) )
        next_states_expanded, next_weights_expanded = sigma_point_expand_JIT( torch.tensor(GA, dtype=torch.float), torch.tensor(PE, dtype=torch.float), gp_params, K_invs, noise, X_s, Y_s, states[i], weights[i], solution, dt_outer, dt_inner) #, gps )       
            
        # Compress back now
        traced_sigma_point_compress_JIT = sigma_point_compress_JIT ### not used # torch.jit.trace( sigma_point_compress_JIT, ( next_states_expanded, next_weights_expanded ) )
        next_states, next_weights = traced_sigma_point_compress_JIT( next_states_expanded, next_weights_expanded )
        
        # Store states and weights
        states.append( next_states ); weights.append( next_weights )
            
        # Get reward 
        traced_reward_UT_Mean_Evaluator_basic = torch.jit.trace( reward_UT_Mean_Evaluator_basic, ( states[i+1], weights[i+1] ) )
        reward = reward + traced_reward_UT_Mean_Evaluator_basic( states[i+1], weights[i+1] )
        
        first_run = False
        print("TRACED!!!!")
    else:
        start_t = 0
        
    for i in range(start_t,H):  
        # Get mean position
        mean_position = traced_get_mean_JIT( states[i], weights[i] )
        
        # Get control input      
        solution = traced_policy( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position )
   
        # Get expanded next state
        next_states_expanded, next_weights_expanded = traced_sigma_point_expand_JIT( torch.tensor(GA, dtype=torch.float), torch.tensor(PE, dtype=torch.float), gp_params, K_invs, noise, X_s, Y_s, states[i], weights[i], solution, dt_outer, dt_inner)#, gps )        
        
        # Compress back now
        next_states, next_weights = traced_sigma_point_compress_JIT( next_states_expanded, next_weights_expanded )
        
        # Store states and weights
        states.append( next_states ); weights.append( next_weights )
            
        # Get reward 
        reward = reward + traced_reward_UT_Mean_Evaluator_basic( states[i+1], weights[i+1] )
        
    return reward

def initialize_gps(noise):
    kern1 = ConstantKernel(constant_value=2.5, constant_value_bounds=(1.0, 1e5)) * RBF(length_scale=np.array([2.0, 2.0, 2.0, 2.0, 2.0]), length_scale_bounds=(0.1, 1e2))
    kern2 = ConstantKernel(constant_value=2.5, constant_value_bounds=(1.0, 1e5)) * RBF(length_scale=np.array([2.0, 2.0, 2.0, 2.0, 2.0]), length_scale_bounds=(0.1, 1e2))
    kern3 = ConstantKernel(constant_value=2.5, constant_value_bounds=(1.0, 1e5)) * RBF(length_scale=np.array([2.0, 2.0, 2.0, 2.0, 2.0]), length_scale_bounds=(0.1, 1e2))
    kern4 = ConstantKernel(constant_value=2.5, constant_value_bounds=(1.0, 1e5)) * RBF(length_scale=np.array([2.0, 2.0, 2.0, 2.0, 2.0]), length_scale_bounds=(0.1, 1e2))
    gp1 = GaussianProcessRegressor(kernel=kern1, alpha = noise, n_restarts_optimizer=10)
    gp2 = GaussianProcessRegressor(kernel=kern2, alpha = noise, n_restarts_optimizer=10)
    gp3 = GaussianProcessRegressor(kernel=kern3, alpha = noise, n_restarts_optimizer=10)
    gp4 = GaussianProcessRegressor(kernel=kern4, alpha = noise, n_restarts_optimizer=10)

    return [gp1, gp2, gp3, gp4]

def train_gps(gps, train_X, train_Y):
    gps[0].fit( train_X, train_Y[:,0] )
    gps[1].fit( train_X, train_Y[:,1] )
    gps[2].fit( train_X, train_Y[:,2] )
    gps[3].fit( train_X, train_Y[:,3] )
    return gps
    
def get_gp_params( gps ):
    param1 = np.exp( gps[0].kernel_.theta )
    param2 = np.exp( gps[1].kernel_.theta )
    param3 = np.exp( gps[2].kernel_.theta )
    param4 = np.exp( gps[3].kernel_.theta )
    print(f"params:{ param1, param2, param3, param4 }")
    return param1, param2, param3, param4

def get_inv_covariances( params1, params2, params3, params4, noise, X_obs ):
    L = 1
    p = 1
    GA = 1; PE = 0
    omega = np.array([[1.0]])
    inv_cov_obs1 = get_covariance_inv_numba( GA, PE, params1[0], params1[1:], L, p, omega, noise, X_obs )
    inv_cov_obs2 = get_covariance_inv_numba( GA, PE, params2[0], params2[1:], L, p, omega, noise, X_obs )
    inv_cov_obs3 = get_covariance_inv_numba( GA, PE, params3[0], params3[1:], L, p, omega, noise, X_obs )
    inv_cov_obs4 = get_covariance_inv_numba( GA, PE, params4[0], params4[1:], L, p, omega, noise, X_obs )
    return inv_cov_obs1, inv_cov_obs2, inv_cov_obs3, inv_cov_obs4
    
def generate_psd_matrices():
    n = 4
    N = 50
    np.random.seed(0)
    B = []
    for i in range(N):
        A = np.random.rand(n,n) 
        A = 0.5 * ( A + A.T )
        A = A + n * np.eye(n)
        if i == 0:
            B = np.copy(A).reshape( (n,n,1) )
        else:
            B = np.append( B, A.reshape( (n,n,1) ), axis = 2 )   
    return B

# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")
env = env_to_render #RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="Excartpole" )
observation, info = env.reset(seed=42)

# Initialize parameters
N = 50
H = 10
param_w = np.random.rand(N)
param_mu = np.random.rand(4,N)
# param_Sigma = torch.rand(10)
param_Sigma = generate_psd_matrices()
lr_rate = 0.05
noise = torch.tensor(0.1, dtype=torch.float)
first_run = True
# X = torch.rand(4).reshape(-1,1)

# Initialize sim parameters
t = 0
dt_inner = 0.02
dt_outer = 0.02
outer_loop = 2
GA = 1.0
PE = 0.0
gps = initialize_gps(noise.item())


train_X = np.array([0,0,0,0, 0]).reshape(-1,1)
train_Y = np.array([0,0,0,0]).reshape(-1,1)

initialize_tensors( env, param_w, param_mu, param_Sigma )

plt.ion()

for i in range(300):
     
    if (i % outer_loop != 0) or i<200:
    
        # Find input
        state = env.get_state()
        state_torch = torch.tensor( state, dtype=torch.float )
        if i<200:
            action = 20*torch.tensor(env.action_space.sample()-0.5)#   2*torch.rand(1)
        else:
            action = policy( env.w_torch, env.mu_torch, env.Sigma_torch, state_torch )
            if (abs(action.item()))>20:
                print("ERROR*************************")
                exit()
        print("action", action)
        observation, reward, terminated, truncated, info = env.step(action.item())
        env.render()
        
        # Get training data fro GP
        state_next = env.get_state()
        state_dot = (state_next - state) / dt_inner
        action_np = np.array([[action.item()]])
        sys_state = np.append( state, action_np, axis =0 )
        if i==0:
            train_X = np.copy( sys_state )
            train_Y = np.copy( state_dot )
        else:
            train_X = np.append( train_X, sys_state, axis = 1 )
            train_Y = np.append( train_Y, state_dot, axis = 1 )
        
        t = t + dt_inner
        
        if terminated or truncated:
            observation, info = env.reset()
        
    else:
        
        initialize_tensors( env, param_w, param_mu, param_Sigma )
        
        psds = generate_psd_matrices()
        
        # Train GP here ##############
        X_s, Y_s = resample_numba(train_X[:,-200:].T, train_Y[:,-200:].T, n_samples = 50)  # sample a subset of training data
        gps = train_gps(gps, X_s, Y_s)
        params1, params2, params3, params4 = get_gp_params( gps )
        cov1, cov2, cov3, cov4 = get_inv_covariances( params1, params2, params3, params4, noise.item(), X_s )
        if np.linalg.det(cov1)>10000:
            print("*************** PROBLEM ******************** ")
        # Plot GP result
        ys1 = []
        covs1 = []
        factor_plot = 2
        for i in range(X_s.shape[0]):
            # t0 = time.time()
            mu1, std1 = gps[0].predict( X_s[i,:].reshape(1,-1)  , return_std=True )
            mu2, std2 = gps[1].predict( X_s[i,:].reshape(1,-1)  , return_std=True )
            mu3, std3 = gps[2].predict( X_s[i,:].reshape(1,-1)  , return_std=True )
            mu4, std4 = gps[3].predict( X_s[i,:].reshape(1,-1)  , return_std=True )
            mu_1 = np.append( mu1.reshape(-1,1), mu2.reshape(-1,1), axis = 1 ); mu_2 =  np.append( mu3.reshape(-1,1), mu4.reshape(-1,1), axis = 1 )
            mu = np.append( mu_1, mu_2, axis = 1 )
            cov_1 = np.append( std1.reshape(-1,1), std2.reshape(-1,1), axis = 1 ); cov_2 = np.append( std3.reshape(-1,1), std4.reshape(-1,1), axis = 1 )
            cov = np.append( cov_1, cov_2, axis = 1 )
            # print("Time taken ", time.time()-t0)
            # print(mu,cov)
            if i==0:#ys == []:
                ys1 = np.copy(mu)
                covs1 = np.copy( cov )
            else:
                ys1 = np.append( ys1, mu, axis=0 )
                covs1 = np.append( covs1, cov , axis=0)
                
        fig1, axis1 = plt.subplots(4,1)
        xx = np.linspace( 1, ys1.shape[0], ys1.shape[0] )
        axis1[0].plot( xx, Y_s[:,0], 'r', label='Actual Value' )
        axis1[0].plot( xx, ys1[:,0], 'g', label='Trained Predicted Mean' )
        axis1[0].fill_between( xx, ys1[:,0] - factor_plot * covs1[:,0], ys1[:,0] + factor_plot * covs1[:,0], color="tab:orange", alpha=0.2 )

        axis1[1].plot( xx, Y_s[:,1], 'r', label='Actual Value' )
        axis1[1].plot( xx, ys1[:,1], 'g', label='Trained Predicted Mean' )
        axis1[1].fill_between( xx, ys1[:,1] - factor_plot * covs1[:,1], ys1[:,1] + factor_plot * covs1[:,1], color="tab:orange", alpha=0.2 )
            
        axis1[2].plot( xx, Y_s[:,2], 'r', label='Actual Value' )
        axis1[2].plot( xx, ys1[:,2], 'g', label='Trained Predicted Mean' )
        axis1[2].fill_between( xx, ys1[:,2] - factor_plot * covs1[:,2], ys1[:,2] + factor_plot * covs1[:,2], color="tab:orange", alpha=0.2 )
            
        axis1[3].plot( xx, Y_s[:,3], 'r', label='Actual Value' )
        axis1[3].plot( xx, ys1[:,3], 'g', label='Trained Predicted Mean' )
        axis1[3].fill_between( xx, ys1[:,3] - factor_plot * covs1[:,3], ys1[:,3] + factor_plot * covs1[:,3], color="tab:orange", alpha=0.2 )
            
            
        # plt.show()
        
        
        
        # params1 = torch.tensor( params1, dtype=torch.float ).reshape(1,-1); params2 = torch.tensor( params2, dtype=torch.float ).reshape(1,-1); params3 = torch.tensor( params3, dtype=torch.float ).reshape(1,-1); params4 = torch.tensor( params4, dtype=torch.float ).reshape(1,-1)
        # params = torch.cat( ( params1, params2, params3, params4 ), dim=0 )
        # cov1 = torch.tensor( cov1, dtype=torch.float ).reshape((X_s.shape[0],X_s.shape[0],1)); cov2 = torch.tensor( cov2, dtype=torch.float ).reshape((X_s.shape[0], X_s.shape[0],1)); cov3 = torch.tensor( cov3, dtype=torch.float ).reshape((X_s.shape[0], X_s.shape[0],1)); cov4 = torch.tensor( cov4, dtype=torch.float ).reshape((X_s.shape[0], X_s.shape[0],1))
        # covs = torch.cat( (cov1, cov2, cov3, cov4), dim = 2 )
        # ##############################
        
        # reward = get_future_reward( env, params, covs, torch.tensor(X_s, dtype=torch.float), torch.tensor(Y_s, dtype=torch.float), noise, gps )        
        # reward.backward(retain_graph = True)
        
        # w_grad = getGrad( env.w_torch )
        # mu_grad = getGrad( env.mu_torch )
        # Sigma_grad = getGrad( env.Sigma_torch )
        
        # param_w = np.clip( param_w - lr_rate * w_grad, -1000.0, 1000.0 )
        # param_mu = np.clip( param_mu - lr_rate * mu_grad, -1000.0, 1000.0 )
        
        
        # print(f"w:{param_w}, mu:{param_mu}")
        # param_Sigma = np.clip( param_Sigma - lr_rate * Sigma_grad, -30.0, 30.0 )
        # Sigma = np.diag( param_Sigma[0:4] )
        # Sigma[0,1] = param_Sigma[4]; Sigma[1,0] = param_Sigma[4]; Sigma[0,2] = param_Sigma[5]; Sigma[2,0] = param_Sigma[5]
        # Sigma[0,3] = param_Sigma[6]; Sigma[3,0] = param_Sigma[6]; Sigma[1,2] = param_Sigma[7]; Sigma[2,1] = param_Sigma[7]
        # Sigma[1,3] = param_Sigma[8]; Sigma[3,1] = param_Sigma[8]; Sigma[2,3] = param_Sigma[9]; Sigma[3,2] = param_Sigma[9]
        # if np.abs(np.linalg.det(Sigma))<0.01:
        #     print("*** ERROR******: wrong covariance matrix")
    
        
# env.close_video_recorder()
env.close()