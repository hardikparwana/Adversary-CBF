import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 27})
import sys
sys.path.append('/home/hardik/Desktop/Research/Adversary-CBF/gpytorch')

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gpytorch

from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from utils.utils import *
from utils.ut_utils import *
from utils.mvgp import *

torch.autograd.set_detect_anomaly(True)


#############################################################
# CBF Controller: centralized
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1 = 1
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 + b1 <= 0]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )
assert cbf_controller.is_dpp()
solver_args = {
            'verbose': False
        }
cbf_controller_layer = CvxpyLayer( cbf_controller, parameters=[ u1_ref, A1, b1 ], variables = [u1] )

###############################################################
def initialize_tensors(follower, leader):
    follower.X_torch = torch.tensor( follower.X, requires_grad = True, dtype=torch.float )
    leader.X_torch = torch.tensor( leader.X, requires_grad = True, dtype=torch.float )
    follower.alpha_torch = torch.tensor(follower.alpha, dtype=torch.float, requires_grad=True)

def compute_A1_b1_tensor(robotsJ, robotsK, alpha, d_min, sys_state):
    h, dh_dxj, dh_dxk = robotsJ.agent_barrier_torch(robotsJ.X_torch, robotsK.X_torch, d_min, robotsK.type)
    A1 = dh_dxj @ robotsJ.g_torch(robotsJ.X_torch)
    
    if robotsK.gp.X == []:
        if robotsK.type=='Unicycle':
            x_dot_k_mean = torch.tensor([0,0,0],dtype=torch.float).reshape(-1,1)
            x_dot_k_cov = torch.eye(3)
        else:
            x_dot_k_mean = torch.tensor([0,0],dtype=torch.float).reshape(-1,1)
    else:             
        x_dot_k_mean, x_dot_k_cov = robotsK.gp.predict_torch( sys_state.T )
        
    x_dot_k = x_dot_k_mean.T.reshape(-1,1) #+ cov terms?? 

    b1 = -dh_dxj @ robotsJ.f_torch(robotsJ.X_torch) - dh_dxk @ x_dot_k - alpha * h      
   
    return A1, b1

def get_follower_input(follower, leader):
    sys_state = torch.cat( (follower.X_torch, leader.X_torch), 0 )
    follower.U_ref_torch = follower.nominal_input_tensor( follower.X_torch, leader.X_torch )
    
    A1, b1 = compute_A1_b1_tensor( follower, leader, follower.alpha_torch, d_min, sys_state )
    
    try:
        solution = cbf_controller_layer( follower.U_ref_torch, A1, b1  )
        return solution[0]
    except Exception as e:
        print(e)
        exit()        
    
def get_future_reward( follower, leader, num_sigma_points ):
    
    # Initialize sigma points for other robots
    follower_states = [torch.clone(follower.X_torch)]        
    prior_leader_states, prior_leader_weights = initialize_sigma_points(leader, num_sigma_points)
    leader_states = [prior_leader_states]
    leader_weights = [prior_leader_weights]

    reward = torch.tensor([0],dtype=torch.float)
  
    for i in range(H):       
        print(f"************** i: {i} ********************")
        # Get sigma points for neighbor's state and state derivative
        leader_xdot_states, leader_xdot_weights = sigma_point_expand( follower_states[i], leader_states[i], leader_weights[i], leader )
        leader_states_expanded, leader_weights_expanded = sigma_point_scale_up( leader_states[i], leader_weights[i])#leader_xdot_weights )
        
        # CBF derivative condition
        A, B = UT_Mean_Evaluator( cbf_condition_evaluator, follower, follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded )
              
        # get nominal controller
        leader_mean_position = get_mean_cov( leader_states[i], leader_weights[i], compute_cov=False )
        u_ref = follower.nominal_input_tensor( follower_states[i], leader_mean_position )
        
        # get CBF solution
        solution,  = cbf_controller_layer( u_ref, A, B )
        print("solution", solution)
        # A.sum().backward(retain_graph=True)
        # print("************* hello *******************")
        # Propagate follower and leader state forward
        follower_states.append( follower.step_torch( follower_states[i], solution, dt_outer ) )
        
        leader_next_state_expanded = leader_states_expanded + leader_xdot_states * dt_outer
        leader_next_states, leader_next_weights = sigma_point_compress( leader_next_state_expanded, leader_xdot_weights )
        leader_states.append( leader_next_states ); leader_weights.append( leader_next_weights )
        
        # Get reward for this state and control input choice = Expected reward in general settings
        reward_function = lambda a, b: follower.compute_reward(a, b, des_d = 0.7)
        reward = reward + UT_Mean_Evaluator_basic( reward_function, follower_states[i+1], leader_states[i+1], leader_weights[i+1])
        # print(f"Reward Checking for Nans: {torch.isnan(reward)}")
        
    # print("forward prop done!")
    return reward
        
################################################################

# Sim Parameters
num_steps = 100
H = 30
outer_loop = 5
t = 0
gp_training_iter = 10
d_min = 0.3
num_points = 5
dt_inner = 0.05
dt_outer = 0.1
alpha_cbf = 0.8   # Initial CBF
num_robots = 1
lr_alpha = 0.05

# Plotting             
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,7),ylim=(-0.5,8))
ax.set_xlabel("X")
ax.set_ylabel("Y")

follower = Unicycle(np.array([0,0,np.pi*0.0]), dt_inner, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf )
leader = SingleIntegrator2D(np.array([1,0]), dt_inner, ax, color='r',palpha=1.0, target = 0)

# Initialize GP
omega = np.array([ [1.0, 0.0],[0.0, 1.0] ])
sigma = 0.2
l = 2.0
leader.gp = gp = MVGP( omega = omega, sigma = sigma, l = l, noise = 0.05, horizon=300 )

max_history = 1000

for i in range(num_steps):

    # High frequency
    if i % outer_loop != 0 or i<10:
        # print("follower alpha fast", follower.alpha)
        # move other agent
        u_leader = np.array([ 1,1 ]).reshape(-1,1)
        leader.step(u_leader, dt_inner)
        
        # implement controller
        initialize_tensors(follower, leader)
        u_follower = get_follower_input(follower, leader)
        # print("u_follower",u_follower)
        follower.step(u_follower.detach().numpy(), dt_inner)
        
        t = t + dt_inner
        
    # Low Frequency tuning
    else: 
        initialize_tensors(follower, leader)
        
        # Train GP
        train_x = np.append( np.copy(follower.Xs[:,-max_history:].T), np.copy(leader.Xs[:,-max_history:].T) , axis = 1  )
        train_y = np.copy(leader.Xdots[:,-max_history:].T)
        # shuffle data??  TODO
        leader.gp.set_XY(train_x, train_y)
        leader.gp.resample( n_samples = 100 )
        leader.gp.train(max_iters=10)
        leader.gp.resample_obs( n_samples = 100 )
        leader.gp.get_obs_covariance()
        gp.initialize_torch()
        
        reward = get_future_reward( follower, leader, num_sigma_points = 1 )

        reward.backward(retain_graph=True)
        
        alpha_grad = getGrad( follower.alpha_torch )
        if abs(alpha_grad)>0.1:
            alpha_grad = np.sign(alpha_grad) * 0.3
        # print("alpha grad", alpha_grad)
        follower.alpha = follower.alpha + lr_alpha * alpha_grad.reshape(-1,1)
        # print("follower alpha", follower.alpha)
        
        exit()

    
    
# Explicit
  
# Step 1

# i = 0
# print(f"************** i: 1 ********************")
# leader_xdot_states1, leader_xdot_weights1 = sigma_point_expand( follower_states[i], leader_states[i], leader_weights[i], leader )
# leader_states_expanded1, leader_weights_expanded1 = sigma_point_scale_up( leader_states[i], leader_xdot_weights1 )

# # CBF derivative condition
# A1, B1 = UT_Mean_Evaluator( cbf_condition_evaluator, follower, follower_states[i], leader_states_expanded1, leader_xdot_states1, leader_weights_expanded1 )
        
# # get nominal controller
# leader_mean_position1 = get_mean_cov( leader_states[i], leader_weights[i], compute_cov=False )
# u_ref1 = follower.nominal_input_tensor( follower_states[i], leader_mean_position1 )

# # get CBF solution
# solution1,  = cbf_controller_layer( u_ref1, A1, B1 )
# # A1.sum().backward(retain_graph=True)
# print("************* hello *******************")
# # Propagate follower and leader state forward
# follower_states.append( follower.step_torch( follower_states[i], solution1, dt_outer ) )

# leader_next_state_expanded1 = leader_states_expanded1 + leader_xdot_states1 * dt_outer
# leader_next_states1, leader_next_weights1 = sigma_point_compress( leader_next_state_expanded1, leader_xdot_weights1 )
# leader_states.append( leader_next_states1 ); leader_weights.append( leader_next_weights1 )

# # Get reward for this state and control input choice = Expected reward in general settings
# reward_function = lambda a, b: follower.compute_reward(a, b, des_d = 0.7)
# reward = reward + UT_Mean_Evaluator_basic( reward_function, follower_states[i+1], leader_states[i+1], leader_weights[i+1])


# # Step 2
# i = 1
# print(f"************** i: 2 ********************")
# leader_xdot_states2, leader_xdot_weights2 = sigma_point_expand( follower_states[i], leader_states[i], leader_weights[i], leader )
# leader_states_expanded2, leader_weights_expanded2 = sigma_point_scale_up( leader_states[i], leader_xdot_weights2 )

# # CBF derivative condition
# A2, B2 = UT_Mean_Evaluator( cbf_condition_evaluator, follower, follower_states[i], leader_states_expanded2, leader_xdot_states2, leader_weights_expanded2 )
        
# # get nominal controller
# leader_mean_position2 = get_mean_cov( leader_states[i], leader_weights[i], compute_cov=False )
# u_ref2 = follower.nominal_input_tensor( follower_states[i], leader_mean_position2 )

# # get CBF solution
# solution2,  = cbf_controller_layer( u_ref2, A2, B2 )
# # A2.sum().backward(retain_graph=True)
# print("************* hello *******************")
# # Propagate follower and leader state forward
# follower_states.append( follower.step_torch( follower_states[i], solution2, dt_outer ) )

# leader_next_state_expanded2 = leader_states_expanded2 + leader_xdot_states2 * dt_outer
# leader_next_states2, leader_next_weights2 = sigma_point_compress( leader_next_state_expanded2, leader_xdot_weights2 )
# leader_states.append( leader_next_states2 ); leader_weights.append( leader_next_weights2 )

# # Get reward for this state and control input choice = Expected reward in general settings
# reward_function = lambda a, b: follower.compute_reward(a, b, des_d = 0.7)
# reward = reward + UT_Mean_Evaluator_basic( reward_function, follower_states[i+1], leader_states[i+1], leader_weights[i+1])

