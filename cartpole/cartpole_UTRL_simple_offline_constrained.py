import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})

import cvxpy as cp
import torch
torch.autograd.set_detect_anomaly(True)

from utils.utils import *
from ut_utils.ut_utilsJIT import *
from utils.mvgp_jit import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from robot_models.custom_cartpole_constrained import CustomCartPoleEnv
from gym_wrappers.record_video import RecordVideo
from cartpole_policy import policy, traced_policy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 

def initialize_tensors(robot, param_w, param_mu, param_Sigma):
    x, x_dot, theta, theta_dot = robot.state
    robot.X_torch = torch.tensor( np.array([ x, x_dot, theta, theta_dot ]).reshape(-1,1), requires_grad = True, dtype=torch.float )
    robot.w_torch = torch.tensor( param_w, requires_grad = True, dtype=torch.float )
    robot.mu_torch = torch.tensor( param_mu, requires_grad = True, dtype=torch.float )
    robot.Sigma_torch = torch.tensor( param_Sigma, requires_grad = True, dtype=torch.float )

x_lim = 1.5
def get_future_reward(robot):
           
    prior_states, prior_weights = initialize_sigma_points_JIT(robot.X_torch)
    states = [prior_states]
    weights = [prior_weights]
    reward = torch.tensor([0],dtype=torch.float)
    
    maintain_constraints = []
    improve_constraints = []    
    
    for i in range(H):  
        # print("hello")
        # Get mean position
        mean_position = traced_get_mean_JIT( states[i], weights[i] )
        
        if np.abs(mean_position[0].detach().numpy()) > 1.5:
            improve_constraints.append( torch.square( mean_position[0] ) )
            print(f"Become Infeasible at :{i}. Need to improve feasibility first")
            return maintain_constraints, improve_constraints, False, reward
        elif torch.square( mean_position[0] ) > x_lim**2 * 5.0 / 6.0:
            maintain_constraints.append( x_lim**2 - torch.square( mean_position[0] ) )
        
        # Get control input      
        solution = traced_policy( robot.w_torch, robot.mu_torch, robot.Sigma_torch, mean_position )
   
        # Get expanded next state
        next_states_expanded, next_weights_expanded = sigma_point_expand_JIT( states[i], weights[i], solution, dt_outer, dt_inner, polemass_length, gravity, length, masspole, total_mass, tau)#, gps )        
        
        # Compress back now
        next_states, next_weights = traced_sigma_point_compress_JIT( next_states_expanded, next_weights_expanded )
        
        # Store states and weights
        states.append( next_states ); weights.append( next_weights )
            
        # Get reward 
        reward = reward + traced_reward_UT_Mean_Evaluator_basic( states[i+1], weights[i+1] )
        
    return maintain_constraints, improve_constraints, True, reward
        
    return reward


def constrained_update( objective, maintain_constraints, improve_constraints, params ) :
    
    num_params = params[0].detach().numpy().size + params[1].detach().numpy().size + params[2].detach().numpy().size
    d = cp.Variable((num_params,1))
    
    # Get Performance optimal direction
    try:
        objective.sum().backward(retain_graph = True) 
        w_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
        mu_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
        Sigma_grad = getGrad(params[2], l_bound = -20.0, u_bound = 20.0 )
        objective_grad = np.append( np.append( w_grad.reshape(1,-1), mu_grad.reshape(1,-1), axis = 1 ), Sigma_grad.reshape(1,-1) , axis = 1)
    except:
        objective_grad = np.zeros( num_params ).reshape(1,-1)
    
    # Get constraint improve direction # assume one at a time
    improve_constraint_direction = np.zeros( num_params ).reshape(1,-1)
    for i, constraint in enumerate( improve_constraints):
        constraint.sum().backward(retain_graph=True)
        w_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
        mu_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
        Sigma_grad = getGrad(params[2], l_bound = -20.0, u_bound = 20.0 )
        improve_constraint_direction = improve_constraint_direction + np.append( np.append( w_grad.reshape(1,-1), mu_grad.reshape(1,-1), axis = 1 ), Sigma_grad.reshape(1,-1) , axis = 1)
    
    # Get allowed directions
    N = len(maintain_constraints)
    if N>0:
        d_maintain = np.zeros((N,num_params))#cp.Variable( (N, num_params) )
        constraints = []
        for i, constraint in enumerate(maintain_constraints):
            constraint.sum().backward(retain_graph=True)
            w_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
            mu_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
            Sigma_grad = getGrad(params[2], l_bound = -20.0, u_bound = 20.0 )
            d_maintain[i,:] = np.append( np.append( w_grad.reshape(1,-1), mu_grad.reshape(1,-1), axis = 1 ), Sigma_grad.reshape(1,-1) , axis = 1)[0]
            
            if constraints ==[]: 
                constraints = constraint.detach().numpy().reshape(-1,1)
            else:
                constraints = np.append( constraints, constraint.detach().numpy().reshape(-1,1), axis = 0 )       

        const = [ constraints + d_maintain @ d >= 0 ]
        const += [ cp.sum_squares( d ) <= 200 ]
        if len(improve_constraint_direction)>0:
            obj = cp.Minimize( improve_constraint_direction @ d )
        else:
            obj = cp.Minimize(  objective_grad @ d  )
        problem = cp.Problem( obj, const )    
        problem.solve( solver = cp.GUROBI )    
        if problem.status != 'optimal':
            print("Cannot Find feasible direction")
            exit()
        
        # print("update direction: ", d.value.T)
        
        return d.value
    
    else:
        if len( improve_constraints ) > 0:
            obj = cp.Maximize( improve_constraint_direction @ d )
            # print("update direction: ", -improve_constraint_direction.reshape(-1,1).T)
            return -improve_constraint_direction.reshape(-1,1)
        else:
            return -objective_grad.reshape(-1,1)

def generate_psd_matrices():
    n = 4
    N = 50
    # np.random.seed(0)
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

def generate_psd_params():
    n = 4
    N = 50
    
    diag = np.random.rand(n) + n
    off_diag = np.random.rand(int( (n**2-n)/2.0 ))
    params = np.append(diag, off_diag, axis = 0).reshape(1,-1)
    
    for i in range(1,50):
        # Diagonal elements
        params_temp = np.random.rand( int(n + (n**2 -n)/2.0) ).reshape(1,-1)
        
        # ## lower Off-diagonal
        # off_diag = np.random.rand(int( (n**2-n)/2.0 ))
        
        # params_temp = np.append(diag, off_diag, axis = 0).reshape(1,-1)
        params = np.append( params, params_temp, axis = 0 )
    
    return params

# Set up environment
env_to_render = CustomCartPoleEnv(render_mode="rgb_array")
# env = env_to_render #RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="Excartpole" )
env = RecordVideo( env_to_render, video_folder="/home/hardik/Desktop/", name_prefix="ExcartpoleSimple_constrained1" )
observation, info = env.reset(seed=42)

polemass_length, gravity, length, masspole, total_mass, tau = torch.tensor(env.polemass_length), torch.tensor(env.gravity), torch.tensor(env.length), torch.tensor(env.masspole), torch.tensor(env.total_mass), torch.tensor(env.tau)

# Initialize parameters
N = 50
H = 20
np.random.seed(0)
param_w = np.random.rand(N) - 0.5#+ 0.5#+ 2.0  #0.5 work with Lr: 5.0
param_mu = np.random.rand(4,N) - 0.5 * np.ones((4,N)) #- 3.5 * np.ones((4,N))
# param_Sigma = torch.rand(10)
# param_Sigma = generate_psd_matrices()
param_Sigma = generate_psd_params()

# param_Sigma = np.random.rand(4,N)

lr_rate = 0.4 #0.1 #0.5#0.001 #0.5
noise = torch.tensor(0.1, dtype=torch.float)
first_run = True
# X = torch.rand(4).reshape(-1,1)

# Initialize sim parameters
t = 0
dt_inner = 0.02
dt_outer = 0.06 # 0.02 # first video with 0.06
outer_loop = 2#4#10 #2


initialize_tensors( env, param_w, param_mu, param_Sigma )

# print(f"s:{param_Sigma}, t:{torch.diag(param_Sigma)}")
# exit()

plt.ion()

for i in range(800): #300
    
    if i==100:
         lr_rate = lr_rate / 2
    elif i == 200:
         lr_rate = lr_rate / 2
     
    # if (i > 400): # (i % outer_loop != 0) or i<1:
    if (i % outer_loop != 0) or i<1:
    
        # Find input
        state = env.get_state()
        state_torch = torch.tensor( state, dtype=torch.float )
       
        action = policy( env.w_torch, env.mu_torch, env.Sigma_torch, state_torch )
        if (abs(action.item()))>20:
            print("ERROR*************************")
            # exit()
        print("action", action)
        observation, reward, terminated, truncated, info = env.step(action.item())
        env.render()
        
        # Get training data fro GP
        # state_next = env.get_state()
        # state_dot = (state_next - state) / dt_inner
        # action_np = np.array([[action.item()]])
        # sys_state = np.append( state, action_np, axis =0 )
        # if i==0:
        #     train_X = np.copy( sys_state )
        #     train_Y = np.copy( state_dot )
        # else:
        #     train_X = np.append( train_X, sys_state, axis = 1 )
        #     train_Y = np.append( train_Y, state_dot, axis = 1 )
        
        t = t + dt_inner
        
        if terminated or truncated:
            observation, info = env.reset()
        
    else:
        
        initialize_tensors( env, param_w, param_mu, param_Sigma )

        success = False
        while not success:
            maintain_constraints, improve_constraints, success, reward = get_future_reward( env ) 
            grads = constrained_update( reward, maintain_constraints, improve_constraints, [env.w_torch, env.mu_torch, env.Sigma_torch] )
            
            grads = np.clip( grads, -2.0, 2.0 )
            param_w = np.clip(param_w + lr_rate * grads[0:param_w.size][:,0], -10, 10 )
            param_mu = np.clip(param_mu + lr_rate * grads[param_w.size:param_w.size + param_mu.size].reshape( 4, 50 ), -10, 10 )
            param_Sigma = np.clip(param_Sigma + lr_rate * grads[param_w.size + param_mu.size:].reshape( 50, 10 ), -1.0, 1.0 )
            # print(f"params w:{param_w}, mu:{param_w}, Sigma:{param_Sigma}")

            initialize_tensors(env, param_w, param_mu, param_Sigma)
        print("Successfully made it feasible")  

        # param_Sigma[ = np.clip( param_Sigma - lr_rate * Sigma_grad, 0.05, 1000 ) 
        
        # print(f"w:{param_w}")#, mu:{param_Sigma}")
        # param_Sigma = np.clip( param_Sigma - lr_rate * Sigma_grad, -30.0, 30.0 )
        # Sigma = np.diag( param_Sigma[0:4] )
        # Sigma[0,1] = param_Sigma[4]; Sigma[1,0] = param_Sigma[4]; Sigma[0,2] = param_Sigma[5]; Sigma[2,0] = param_Sigma[5]
        # Sigma[0,3] = param_Sigma[6]; Sigma[3,0] = param_Sigma[6]; Sigma[1,2] = param_Sigma[7]; Sigma[2,1] = param_Sigma[7]
        # Sigma[1,3] = param_Sigma[8]; Sigma[3,1] = param_Sigma[8]; Sigma[2,3] = param_Sigma[9]; Sigma[3,2] = param_Sigma[9]
        # if np.abs(np.linalg.det(Sigma))<0.01:
        #     print("*** ERROR******: wrong covariance matrix")
    
        
# env.close_video_recorder()
env.close()