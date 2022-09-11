import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

import torch
torch.autograd.set_detect_anomaly(True)

from utils.utils import *
from utils.ut_utilsJIT import *
from utils.mvgp_jit import *

from robot_models.custom_cartpole import CustomCartPoleEnv
from gym_wrappers.record_video import RecordVideo
from cartpole_policy import policy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 

def initialize_tensors(robot, param_w, param_mu, param_Sigma):
    x, x_dot, theta, theta_dot = robot.state
    robot.X_torch = torch.tensor( [ x, x_dot, theta, theta_dot ], requires_grad = True, dtype=torch.float ).reshape(-1,1)
    robot.w_torch = torch.tensor( param_w, requires_grad = True, dtype=torch.float )
    robot.mu_torch = torch.tensor( param_mu, requires_grad = True, dtype=torch.float )
    robot.Sigma_torch = torch.tensor( param_Sigma, requires_grad = True, dtype=torch.float )
    
# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="human")
env = env_to_render #RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="Excartpole" )
observation, info = env.reset(seed=42)

# Initialize parameters
N = 50
param_w = torch.rand(N)
param_mu = torch.rand((4,N))
param_Sigma = torch.rand(10)
lr_rate = 0.05
# X = torch.rand(4).reshape(-1,1)

# Initialize sim parameters
t = 0
dt_inner = 0.02
outer_loop = 2

for i in range(100):
    
    if (i % outer_loop != 0):
    
        # Find input
        state = env.get_state()
        state_torch = torch.tensor( state, dtype=torch.float )
        action = policy( param_w, param_mu, param_Sigma, state_torch )
        observation, reward, terminated, truncated, info = env.step(action.item())
        env.render()
        
        t = t + dt_inner
        
        if terminated or truncated:
            observation, info = env.reset()
        
    else:
        
        initialize_tensors( env, param_w, param_mu, param_Sigma )
        
        # Train GP here ##############
        
        ##############################
        
        reward = get_future_reward( env  )        
        reward.backward(retain_graph = True)
        
        w_grad = getGrad( env.w_torch )
        mu_grad = getGrad( env.mu_torch )
        Sigma_grad = getGrad( env.Sigma_torch )
        
        param_w = np.clip( param_w - lr_rate * w_grad, -1000.0, 1000.0 )
        param_mu = np.clip( param_mu - lr_rate * mu_grad, -1000.0, 1000.0 )
        param_Sigma = np.clip( param_Sigma - lr_rate * Sigma_grad, -1000.0, 1000.0 )
    
        
# env.close_video_recorder()
env.close()