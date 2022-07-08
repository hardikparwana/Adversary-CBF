from cProfile import label
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from utils.utils import *

from matplotlib.animation import FFMpegWriter
from cvxpylayers.torch import CvxpyLayer

plt.rcParams.update({'font.size': 27})

# Sim Parameters                  
dt = 0.05
tf = 5.4#8#4.1 #0.2#4.1
num_steps = int(tf/dt)
outer_loop = 1
dt_outer = dt * outer_loop
t = 0
d_min = 1.0#0.1

h_min = 1.0##0.4   # more than this and do not decrease alpha
min_dist = 1.0 # 0.1#0.05  # less than this and dercrease alpha
cbf_extra_bad = 0.0
update_param = True

alpha_cbf = 0.8
alpha_der_max = 0.5#1.0#0.5

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,7),ylim=(-0.5,8))
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_aspect(1)


num_adversaries = 3
alpha = 0.1

default_plot = False
save_plot = False
movie_name = 'test0_default.mp4'

# agents
robots = []
num_robots = 3
robots.append( Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries ) )
robots.append( Unicycle(np.array([2.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries ) )
robots.append( Unicycle(np.array([3.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries ) )

# agent nominal version
robots_nominal = []

robots_nominal.append( Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=alpha) )
robots_nominal.append( Unicycle(np.array([2.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=alpha ) )
robots_nominal.append( Unicycle(np.array([3.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=alpha ) )
U_nominal = np.zeros((2,num_robots))

# Uncooperative
greedy = []
greedy.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=1.0) )
greedy.append( SingleIntegrator2D(np.array([0,5]), dt, ax, color='r',palpha=1.0) )
greedy.append( SingleIntegrator2D(np.array([7,7]), dt, ax, color='r',palpha=1.0) )

greedy_nominal = []
greedy_nominal.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=alpha) )
greedy_nominal.append( SingleIntegrator2D(np.array([0,5]), dt, ax, color='r',palpha=alpha) )
greedy_nominal.append( SingleIntegrator2D(np.array([7,7]), dt, ax, color='r',palpha=alpha) )


# Adversarial agents
# adversary = []
# adversary.append( SingleIntegrator2D(np.array([0,4]), dt, ax) )


############################## Optimization problems ######################################

###### 1: CBF Controller: centralized
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1 + num_adversaries
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 <= b1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller_step1 = cp.Problem( objective1, const1 )
assert cbf_controller_step1.is_dpp()
solver_args = {
            'verbose': False
        }
cbf_controller_step1_layer = CvxpyLayer( cbf_controller_step1, parameters=[ u1_ref, A1, b1 ], variables = [u1] )
# torch version
# use torch for this rather than doing myself you fool


###### 2: CBF controller: centralized
u2 = cp.Variable( (2,1) )
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints2 = num_robots - 1 + num_adversaries
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 <= b2]
# const2 += [cp.abs(u2[0,0])<=10.0]
# const2 += [cp.abs(u2[1,0])<=40.0]
# Q2 = cp.Parameter( (1,2), value = np.zeros((1,2)) )
# objective2 = cp.Minimize( Q2 @ u2 )
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) )
cbf_controller_step2 = cp.Problem( objective2, const2 )

##########################################################################################
      
# for i in range(num_steps):
    
#     const_index = 0
    
#     ## Greedy's nominal movement
#     u_greedy_nominal = np.array([1.0, 0.0])
#     greedy_nominal[0].step(u_greedy_nominal)
    
#     ## Greedy's believed movement
#     V_nominal, dV_dx_nominal = greedy[0].lyapunov( greedy_nominal[0].X  )
#     greedy[0].x_dot_nominal = -1.0 * dV_dx_nominal.T /np.linalg.norm( dV_dx_nominal )
    
#     ## Greedy actual movement
#     V, dV_dx = greedy[0].lyapunov( robots[0].X )
#     greedy[0].U_ref = -1.0 * dV_dx.T / np.linalg.norm( dV_dx )
        
#     # Move nominal agents
#     for j in range(num_robots):
#         u_nominal = np.array([1.0,-0.5])
#         robots_nominal[j].step( u_nominal )
#         V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
#         robots[j].x_dot_nominal = -3.0*dV_dx.T/np.linalg.norm(dV_dx)
#         robots[j].U_ref = robots[j].nominal_input( robots_nominal[j] )
#         print(f" {j} input: {robots[j].U_ref} ")
#         robots_nominal[j].render_plot()
        
#     t = t + dt
    
#     fig.canvas.draw()
#     fig.canvas.flush_events()
    
    
# exit()
tp = [] 

for i in range(num_steps):
    
    ## Make copies of current state
    for j in range(num_adversaries):
        greedy[j].X_org = np.copy(greedy[j].X)
        
    for j in range(num_robots):
        robots[j].X_org = np.copy(robots[j].X)
        
    ## Low frequency operation
    if num_steps % outer_loop != 0:
        print("High frequency update")
         # higher-frequency loop
    
    else:
        
        
        ######################################### STEP 1 #################################################
        const_index = 0
        
        # Greedy agents
        for j in range(num_adversaries):
            ## Greedy's nominal movement
            if j==0 or j==1:
                u_greedy_nominal = np.array([1.0, 0.0])
            else:
                u_greedy_nominal = np.array([-1.0,0.0])
            greedy_nominal[j].step(u_greedy_nominal, dt_outer)
            
            ## Greedy's believed movement
            V_nominal, dV_dx_nominal = greedy[j].lyapunov( greedy_nominal[j].X  )
            greedy[j].x_dot_nominal = -1.0 * dV_dx_nominal.T /np.linalg.norm( dV_dx_nominal )
            
            ## Greedy actual movement
            if j==0:
                V, dV_dx = greedy[0].lyapunov( robots[0].X )
                greedy[j].U_ref = -1.0 * dV_dx.T / np.linalg.norm( dV_dx )
            else:
                greedy[j].U_ref = u_greedy_nominal
                
        # Move nominal agents
        for j in range(num_robots):
            u_nominal = np.array([1.0,0.0])
            robots_nominal[j].step( u_nominal, dt_outer )
            V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
            robots[j].x_dot_nominal = -3.0*dV_dx.T/np.linalg.norm(dV_dx)
            robots[j].U_ref = robots[j].nominal_input( robots_nominal[j] )
            robots_nominal[j].render_plot()
        
        for j in range(num_robots):
            
            const_index = 0
                            
            # greedy
            for k in range(num_adversaries):
                h, dh_dxi, dh_dxk, dh_dxi_dxi, dh_dxi_dxk, dh_dxk_dxi, dh_dxk_dxj = robots[j].agent_barrier(greedy[k], d_min);  
                robots[j].adv_h[0,k] = h
                
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - dh_dxk @ ( greedy[k].f() + greedy[k].g() @ greedy[k].U ) - cbf_extra_bad - robots[j].adv_alpha[0,k] * h           
                
                # dA1_dxi = 
                
                
                const_index = const_index + 1
                
            for k in range(num_robots):
                
                if k==j:
                    continue
                
                h, dh_dxj, dh_dxk, dh_dxi_dxi, dh_dxi_dxk, dh_dxk_dxi, dh_dxk_dxj = robots[j].agent_barrier(robots[k], d_min);
                robots[j].robot_h[0,k] = h
                    
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxj @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxj @ robots[j].f() - dh_dxk @ ( robots[k].f() + robots[k].g() @ robots[k].U ) - cbf_extra_bad - robots[j].robot_alpha[0,k] * h

                const_index = const_index + 1
                
            
        # Solve step1 control inputk
        for j in range(num_robots):
            
            const_index = 0      
            # Constraints in LP and QP are same      
            A1.value = robots[j].A1
            b1.value = robots[j].b1
                    
            # Solve for control input
            u1_ref.value = robots[j].U_ref
            cbf_controller_step1.solve(requires_grad=True)  #cbf_controller.solve(solver=cp.GUROBI, requires_grad=True)
            if cbf_controller_step1.status!='optimal':
                print(f"{j}'s input: {cbf_controller_step1.status}")
            robots[j].nextU = u1.value       
            cbf_controller_step1.backward()
            A1grad_sum = A1.gradient
            b1grad_sum = b1.gradient 
            
            A1.value = robots[j].A1 * np.concatenate( ( np.ones(( np.shape(robots[j].A1)[0],1 )), -1* np.ones(( np.shape(robots[j].A1)[0],1 )) ), axis=1 )  # N x 2
            u1_ref.value = robots[j].U_ref * np.array([1,-1]).reshape(-1,1)    
            cbf_controller_step1.solve(requires_grad=True)  
            cbf_controller_step1.backward()
            A1grad_diff = A1.gradient
            b1grad_diff = b1.gradient 
            A1grad
                    
        for j in range(num_adversaries):
            ## Save original control input
            greedy[j].U_org = greedy[j].U_ref
            
            greedy[j].step(greedy[j].U_ref, dt_outer)
            greedy[j].render_plot()    
        
        for j in range(num_robots):
            ## Save original control input
            robots[j].U_org = robots[j].nextU
            
            robots[j].step( robots[j].nextU, dt_outer )
            robots[j].render_plot()
            # print(f"{j} state: {robots[j].X[1,0]}, input:{robots[j].nextU[0,0]}, {robots[j].nextU[1,0]}")
            
        ######################################### STEP 2 #################################################
        
        const_index = 0
        
        # Greedy agents
        for j in range(num_adversaries):
            ## Greedy's nominal movement
            if j==0 or j==1:
                u_greedy_nominal = np.array([1.0, 0.0])
            else:
                u_greedy_nominal = np.array([-1.0,0.0])
            greedy_nominal[j].step(u_greedy_nominal, dt_outer)
            
            ## Greedy's believed movement
            V_nominal, dV_dx_nominal = greedy[j].lyapunov( greedy_nominal[j].X  )
            greedy[j].x_dot_nominal = -1.0 * dV_dx_nominal.T /np.linalg.norm( dV_dx_nominal )
            
            ## Greedy actual movement
            if j==0:
                V, dV_dx = greedy[0].lyapunov( robots[0].X )
                greedy[j].U_ref = -1.0 * dV_dx.T / np.linalg.norm( dV_dx )
            else:
                greedy[j].U_ref = u_greedy_nominal
                
        # Move nominal agents
        for j in range(num_robots):
            u_nominal = np.array([1.0,0.0])
            robots_nominal[j].step( u_nominal, dt_outer )
            V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
            robots[j].x_dot_nominal = -3.0*dV_dx.T/np.linalg.norm(dV_dx)
            robots[j].U_ref = robots[j].nominal_input( robots_nominal[j] )
            robots_nominal[j].render_plot()
        
        for j in range(num_robots):
            
            const_index = 0
                            
            # greedy
            for k in range(num_adversaries):
                h, dh_dxi, dh_dxk, dh_dxi_dxi, dh_dxi_dxk, dh_dxk_dxi, dh_dxk_dxj = robots[j].agent_barrier(greedy[k], d_min);  
                robots[j].adv_h[0,k] = h
                
                # Control QP constraint
                robots[j].A2[const_index,:] = dh_dxi @ robots[j].g()
                robots[j].b2[const_index] = -dh_dxi @ robots[j].f() - dh_dxk @ ( greedy[k].f() + greedy[k].g() @ greedy[k].U ) - cbf_extra_bad - robots[j].adv_alpha[0,k] * h           
                const_index = const_index + 1
                
            for k in range(num_robots):
                
                if k==j:
                    continue 
                
                h, dh_dxj, dh_dxk, dh_dxi_dxi, dh_dxi_dxk, dh_dxk_dxi, dh_dxk_dxj = robots[j].agent_barrier(robots[k], d_min);
                robots[j].robot_h[0,k] = h
                    
                # Control QP constraint
                robots[j].A2[const_index,:] = dh_dxj @ robots[j].g()
                robots[j].b2[const_index] = -dh_dxj @ robots[j].f() - dh_dxk @ ( robots[k].f() + robots[k].g() @ robots[k].U ) - cbf_extra_bad - robots[j].robot_alpha[0,k] * h

                const_index = const_index + 1
                
            
        # Solve step2 control inputk
        for j in range(num_robots):
            
            const_index = 0      
            # Constraints in LP and QP are same      
            A2.value = robots[j].A2
            b2.value = robots[j].b2
                    
            # Solve for control input
            u2_ref.value = robots[j].U_ref
            cbf_controller_step2.solve(requires_grad=True)  #cbf_controller.solve(solver=cp.GUROBI, requires_grad=True)
            if cbf_controller_step2.status!='optimal':
                print(f"{j}'s input: {cbf_controller_step2.status}")
            robots[j].nextU = u2.value       
            
        for j in range(num_adversaries):
            greedy[j].step(greedy[j].U_ref, dt_outer)
            greedy[j].render_plot()    
        
        for j in range(num_robots):
            robots[j].step( robots[j].nextU, dt_outer )
            robots[j].render_plot()
            # print(f"{j} state: {robots[j].X[1,0]}, input:{robots[j].nextU[0,0]}, {robots[j].nextU[1,0]}")   
            
        ############################# Calculate gradients ################################################
        
        
        

        ##################################### Actual state update #############################################
        for j in range(num_adversaries):
            ## Save original control input
            greedy[j].X = greedy[j].X_org            
            greedy[j].step(greedy[j].U_org, dt)
            greedy[j].render_plot()    
        
        for j in range(num_robots):
            ## Save original control input
            robots[j].X = robots[j].X_org            
            robots[j].step( robots[j].U_org, dt )
            robots[j].render_plot()
            # print(f"{j} state: {robots[j].X[1,0]}, input:{robots[j].nextU[0,0]}, {robots[j].nextU[1,0]}")
            
            #####################################################################################################
    
    t = t + dt
    tp.append(t)
    
    fig.canvas.draw()
    fig.canvas.flush_events()

    
plt.ioff()   

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# matplotlib.rc('font', **font)
