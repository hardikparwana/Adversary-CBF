import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 14})
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
delta = cp.Variable((4,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1 = 1 + 3
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 + b1 + delta >= 0]
const1 += [ delta[1] == 0 ]
const1 += [ delta[2] == 0 ]
const1 += [ delta[3] == 0 ]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) + 100*cp.sum_squares(delta) )
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
    follower.k_torch = torch.tensor( follower.k, dtype=torch.float, requires_grad = True )

# def compute_A1_b1_tensor(robotsJ, robotsK, alpha, sys_state):
#     h, dh_dxj, dh_dxk = robotsJ.agent_barrier_torch(robotsJ.X_torch, robotsK.X_torch, robotsK.type)
#     A1 = dh_dxj @ robotsJ.g_torch(robotsJ.X_torch)
    
#     if robotsK.gp.X == []:
#         if robotsK.type=='Unicycle':
#             x_dot_k_mean = torch.tensor([0,0,0],dtype=torch.float).reshape(-1,1)
#             x_dot_k_cov = torch.eye(3)
#         else:
#             x_dot_k_mean = torch.tensor([0,0],dtype=torch.float).reshape(-1,1)
#     else:             
#         x_dot_k_mean, x_dot_k_cov = robotsK.gp.predict_torch( sys_state.T )
#         # print(f"gp mean: { x_dot_k_mean }, actual_last_xdot: {robotsK.Xdots[:,-1]}")
        
#     x_dot_k = x_dot_k_mean.T.reshape(-1,1) #+ cov terms?? 
#     # if robotsK.Xdots == []:
#     #     x_dot_k = torch.zeros(2, type=torch.float).reshape(-1,1)
#     # else: 
#     #     x_dot_k = torch.tensor(robotsK.Xdots[:,-1], dtype=torch.float).reshape(-1,1)

#     b1 = dh_dxj @ robotsJ.f_torch(robotsJ.X_torch) + dh_dxk @ x_dot_k + alpha * h      
   
#     return A1, b1

def compute_A1_b1_tensor(robotsJ, robotsK, alpha, sys_state):
    
    if robotsK.gp.X == []:
        if robotsK.type=='Unicycle':
            x_dot_k_mean = torch.tensor([0,0,0],dtype=torch.float).reshape(-1,1)
            x_dot_k_cov = torch.eye(3)
        else:
            x_dot_k_mean = torch.tensor([0,0],dtype=torch.float).reshape(-1,1)
    else:             
        x_dot_k_mean, x_dot_k_cov = robotsK.gp.predict_torch( sys_state.T )
        # print(f"gp mean: { x_dot_k_mean }, actual_last_xdot: {robotsK.Xdots[:,-1]}")
        
    x_dot_k = x_dot_k_mean.T.reshape(-1,1) #+ cov terms?? 
    
    A1, b1 = clf_cbf_fov_evaluator(robotsJ, robotsJ.X_torch, robotsK.X_torch, x_dot_k, robotK_type='SingleIntegrator2D')
   
    return A1, b1

def get_follower_input(follower, leader):
    # sys_state = torch.cat( (follower.X_torch, leader.X_torch), 0 )
    sys_state = leader.X_torch
    follower.U_ref_torch = follower.nominal_input_tensor( follower.X_torch, leader.X_torch )
    
    A1, b1 = compute_A1_b1_tensor( follower, leader, follower.alpha_torch, sys_state )
    
    try:
        solution,  = cbf_controller_layer( follower.U_ref_torch, A1, b1  )
        # solution = follower.U_ref_torch 
        # print(f"U_ref:{ follower.U_ref_torch.reshape(1,-1) }, cbf: {solution.reshape(1,-1)}")
        return solution
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
        # print(f"************** i: {i} ********************")
        # Get sigma points for neighbor's state and state derivative
        leader_xdot_states, leader_xdot_weights = sigma_point_expand( follower_states[i], leader_states[i], leader_weights[i], leader )
        leader_states_expanded, leader_weights_expanded = sigma_point_scale_up( leader_states[i], leader_weights[i])#leader_xdot_weights )
        
        # CBF derivative condition
        # A, B = UT_Mean_Evaluator( cbf_condition_evaluator, follower, follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded )
        A, B = UT_Mean_Evaluator( clf_cbf_fov_evaluator, follower, follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded )
              
        # get nominal controller
        leader_mean_position = get_mean_cov( leader_states[i], leader_weights[i], compute_cov=False )
        u_ref = follower.nominal_input_tensor( follower_states[i], leader_mean_position )
        
        # get CBF solution
        solution,  = cbf_controller_layer( u_ref, A, B )
        # solution = u_ref
        # print("solution", solution)
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
num_steps = 150 #200
learn_period = 50
H = 5
outer_loop = 5
gp_training_iter = 10
d_min = 0.3
d_max = 2.0
angle_max = np.pi/2
num_points = 5
dt_inner = 0.05
dt_outer = 0.1
alpha_cbf = 0.8   # Initial CBF
num_robots = 1
lr_alpha = 0.05

# Plotting             
plt.ion()

gp_training_iter_init = 30

# Without adapt
t = 0
first_time = True
fig = plt.figure()
ax = plt.axes(xlim=(0,10),ylim=(-5,8))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect(1)

follower = Unicycle(np.array([0,0,np.pi*0.0]), dt_inner, ax, num_robots=num_robots, id = 0, min_D = d_min, max_D = d_max, FoV_angle = angle_max, color='g',palpha=1.0, alpha=alpha_cbf, num_alpha = 3 )
leader = SingleIntegrator2D(np.array([1,0]), dt_inner, ax, color='r',palpha=1.0, target = 0)

# Initialize GP
omega = np.array([ [1.0, 0.0],[0.0, 1.0] ])
sigma = 0.2
l = 2.0
leader.gp = gp = MVGP( omega = omega, sigma = sigma, l = l, noise = 0.05, horizon=300 )

max_history = 100


step_rewards = []
gp_pred_x = []
gp_pred_y = []
true_x = []
true_y = []
gp_pred_x_cov = []
gp_pred_y_cov = []

for i in range(num_steps):

    # High frequency
    if i % outer_loop != 0 or i<learn_period:
        # print("follower alpha fast", follower.alpha)
        # move other agent
        # u_leader = np.array([ 1,1 ]).reshape(-1,1)
        
        # uL = 0.5
        # vL = 3*np.sin(np.pi*t*4) #  0.1 # 1.2
        uL = 1
        vL = 1
        
        
        u_leader = np.array([ uL, vL ]).reshape(-1,1)
        
        leader.step(u_leader, dt_inner)
        
        # implement controller
        initialize_tensors(follower, leader)
        u_follower = get_follower_input(follower, leader)
        # print("u_follower",u_follower)
        follower.step(u_follower.detach().numpy(), dt_inner)
        
        step_rewards.append( follower.lyapunov(follower.X, leader.X) )
        
        t = t + dt_inner
        
    # Low Frequency tuning
    else: 
        initialize_tensors(follower, leader)
        
        # Train GP
        # train_x = np.append( np.copy(follower.Xs[:,-max_history:].T), np.copy(leader.Xs[:,-max_history:].T) , axis = 1  )
        train_x = np.copy(leader.Xs[:,-max_history:]).T
        train_y = np.copy(leader.Xdots[:,-max_history:].T)
        # shuffle data??  TODO
        leader.gp.set_XY(train_x, train_y)
        leader.gp.resample( n_samples = 50 )
        if first_time:
            leader.gp.train(max_iters=gp_training_iter_init, n_samples = 50, print_status = True)
            first_time = False
        else:
            leader.gp.train(max_iters=10, n_samples = 50, print_status = True)
        leader.gp.resample_obs( n_samples = 50 )
        leader.gp.get_obs_covariance()
        gp.initialize_torch()
        
        true_x.append(uL)
        true_y.append(vL)
        mu, cov = leader.gp.predict( leader.X.T )
        gp_pred_x.append( mu[0,0] )
        gp_pred_y.append( mu[0,1] )
        gp_pred_x_cov.append( np.sqrt(cov[0,0]) )
        gp_pred_y_cov.append( np.sqrt(cov[1,1]) )
        
        # reward = get_future_reward( follower, leader, num_sigma_points = 1 )

        # reward.backward(retain_graph=True)
        
        # # Get grads
        # alpha_grad = getGrad( follower.alpha_torch )
        # alpha_grad = np.clip( alpha_grad, -0.1, 0.1 )
                
        # k_grad = getGrad( follower.k_torch )
        # k_grad = np.clip( k_grad, -0.1, 0.1 )
        
        # print(f"grads: alpha:{ alpha_grad.T }, k:{ k_grad }")
        # # if abs(alpha_grad)>0.1:
        # #     alpha_grad = np.sign(alpha_grad) * 0.3
        # # print("alpha grad", alpha_grad)
        # follower.alpha = follower.alpha + lr_alpha * alpha_grad.reshape(-1,1)
        # follower.k = follower.k + lr_alpha * k_grad
        # # print("follower alpha", follower.alpha)
        
        # exit()
        
    fig.canvas.draw()
    fig.canvas.flush_events()

# With adapt
t = 0
first_time = True
fig = plt.figure()
ax = plt.axes(xlim=(0,10),ylim=(-5,5))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect(1)

follower = Unicycle(np.array([0,0,np.pi*0.0]), dt_inner, ax, num_robots=num_robots, id = 0, min_D = d_min, max_D = d_max, FoV_angle = angle_max, color='g',palpha=1.0, alpha=alpha_cbf, num_alpha = 3 )
leader = SingleIntegrator2D(np.array([1,0]), dt_inner, ax, color='r',palpha=1.0, target = 0)

# Initialize GP
omega = np.array([ [1.0, 0.0],[0.0, 1.0] ])
sigma = 0.2
l = 2.0
leader.gp = gp = MVGP( omega = omega, sigma = sigma, l = l, noise = 0.05, horizon=300 )

max_history = 100

step_rewards_adapt = []
gp_pred_x_adapt = []
gp_pred_y_adapt = []
true_x_adapt = []
true_y_adapt = []
gp_pred_x_cov_adapt = []
gp_pred_y_cov_adapt = []

for i in range(num_steps):

    # High frequency
    if i % outer_loop != 0 or i<learn_period:
        # print("follower alpha fast", follower.alpha)
        # move other agent
        # u_leader = np.array([ 1,1 ]).reshape(-1,1)
        uL = 1
        vL = 1
        # uL = 0.5
        # vL = 3*np.sin(np.pi*t*4) #  0.1 # 1.2
        u_leader = np.array([ uL, vL ]).reshape(-1,1)
        
        leader.step(u_leader, dt_inner)
        
        # implement controller
        initialize_tensors(follower, leader)
        u_follower = get_follower_input(follower, leader)
        # print("u_follower",u_follower)
        follower.step(u_follower.detach().numpy(), dt_inner)
        
        step_rewards_adapt.append( follower.lyapunov(follower.X, leader.X) )
        
        t = t + dt_inner
        
    # Low Frequency tuning
    else: 
        initialize_tensors(follower, leader)
        
        # Train GP
        # train_x = np.append( np.copy(follower.Xs[:,-max_history:].T), np.copy(leader.Xs[:,-max_history:].T) , axis = 1  )
        train_x = np.copy(leader.Xs[:,-max_history:]).T
        train_y = np.copy(leader.Xdots[:,-max_history:].T)
        # shuffle data??  TODO
        leader.gp.set_XY(train_x, train_y)
        leader.gp.resample( n_samples = 50 )
        if first_time:
            leader.gp.train(max_iters=gp_training_iter_init, n_samples = 50, print_status = True)
            first_time = False
        else:
            leader.gp.train(max_iters=10, n_samples = 50, print_status = True)

        leader.gp.resample_obs( n_samples = 50 )
        leader.gp.get_obs_covariance()
        gp.initialize_torch()
        
        true_x_adapt.append(uL)
        true_y_adapt.append(vL)
        mu, cov = leader.gp.predict( leader.X.T )
        gp_pred_x_adapt.append( mu[0,0] )
        gp_pred_y_adapt.append( mu[0,1] )
        gp_pred_x_cov_adapt.append( np.sqrt(cov[0,0]) )
        gp_pred_y_cov_adapt.append( np.sqrt(cov[1,1]) )
        
        reward = get_future_reward( follower, leader, num_sigma_points = 1 )

        reward.backward(retain_graph=True)
        
        # Get grads
        alpha_grad = getGrad( follower.alpha_torch )
        alpha_grad = np.clip( alpha_grad, -0.1, 0.1 )
                
        k_grad = getGrad( follower.k_torch )
        k_grad = np.clip( k_grad, -0.1, 0.1 )
        
        print(f"grads: alpha:{ alpha_grad.T }, k:{ k_grad }")
        # if abs(alpha_grad)>0.1:
        #     alpha_grad = np.sign(alpha_grad) * 0.3
        # print("alpha grad", alpha_grad)
        follower.alpha = follower.alpha - lr_alpha * alpha_grad.reshape(-1,1)
        follower.k = follower.k - lr_alpha * k_grad
        # print("follower alpha", follower.alpha)
        
        # exit()
        
    fig.canvas.draw()
    fig.canvas.flush_events()
  
plt.ioff()

true_x = np.asarray(true_x)
true_y = np.asarray(true_y)
gp_pred_x = np.asarray(gp_pred_x)
gp_pred_y = np.asarray(gp_pred_y)
gp_pred_x_cov = np.asarray(gp_pred_x_cov)
gp_pred_y_cov = np.asarray(gp_pred_y_cov)
 
gp_pred_x_adapt = np.asarray(gp_pred_x_adapt)
gp_pred_y_adapt = np.asarray(gp_pred_y_adapt)
gp_pred_x_cov_adapt = np.asarray(gp_pred_x_cov_adapt)
gp_pred_y_cov_adapt = np.asarray(gp_pred_y_cov_adapt)

factor = 1.96
 
fig1, axis1 = plt.subplots(1,1)
axis1.plot(step_rewards, 'r', label='Without Adaptation')
axis1.plot(step_rewards_adapt, 'g', label='With Adaptation')
axis1.title.set_text("Step Reward")
    
fig2, axis2 = plt.subplots(2,1)
xx = np.linspace(1,len(true_x),len(true_x))
axis2[0].plot(xx, true_x, 'r', label='Without Adaptation')
axis2[0].plot(xx, gp_pred_x, 'g', label='With Adaptation')
axis2[0].title.set_text("X dot Prediction: No Adaptation")
axis2[0].fill_between( xx, gp_pred_x - factor * gp_pred_x_cov, gp_pred_x + factor * gp_pred_x_cov, color="tab:orange", alpha = 0.2 )

axis2[1].plot(xx, true_y, 'r', label='Without Adaptation')
axis2[1].plot(xx, gp_pred_y, 'g', label='Without Adaptation')
axis2[1].title.set_text("Y dot Prediction: No Adaptation")
axis2[1].fill_between( xx, gp_pred_y - factor * gp_pred_y_cov, gp_pred_y + factor * gp_pred_y_cov, color="tab:orange", alpha = 0.2 )

fig3, axis3 = plt.subplots(2,1)
axis3[0].plot(xx, true_x_adapt, 'r', label='Without Adaptation')
axis3[0].plot(xx, gp_pred_x_adapt, 'g', label='With Adaptation')
axis3[0].title.set_text("Y dot Prediction: With Adaptation")
axis3[0].fill_between( xx, gp_pred_x_adapt - factor * gp_pred_x_cov_adapt, gp_pred_x_adapt + factor * gp_pred_x_cov_adapt, color="tab:orange", alpha = 0.2 )

axis3[1].plot(xx, true_y_adapt, 'r', label='Without Adaptation')
axis3[1].plot(xx, gp_pred_y_adapt, 'g', label='With Adaptation')
axis3[1].title.set_text("Y dot Prediction: With Adaptation")
axis3[1].fill_between( xx, gp_pred_y_adapt - factor * gp_pred_y_cov_adapt, gp_pred_y_adapt + factor * gp_pred_y_cov_adapt, color="tab:orange", alpha = 0.2 )

plt.show()
    
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

