import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from utils.utils import *

# Sim Parameters                  
dt = 0.05
tf = 10
num_steps = int(tf/dt)
t = 0
d_min = 0.1
h_min = 0.4

min_dist = 0.05
alpha_cbf = 0.8
alpha_der_max = 0.5

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,10),ylim=(-10,10))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect(1)

# agents
robots = []
num_robots = 3
robots.append( SingleIntegrator2D(np.array([3,1]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0 ) )
robots.append( SingleIntegrator2D(np.array([2.5,0]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0 ) )
robots.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0 ) )

# agent nominal version
robots_nominal = []
num_robots = 3
robots_nominal.append( SingleIntegrator2D(np.array([3,1]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=0.4 ) )
robots_nominal.append( SingleIntegrator2D(np.array([2.5,0]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=0.4 ) )
robots_nominal.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=0.4 ) )
U_nominal = np.zeros((2,num_robots))

# Uncooperative
greedy = []
greedy.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=1.0) )

greedy_nominal = []
greedy_nominal.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=0.4) )

# Adversarial agents
# adversary = []
# adversary.append( SingleIntegrator2D(np.array([0,4]), dt, ax) )
num_adversaries = 1

############################## Optimization problems ######################################

###### 1: CBF Controller
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1 + num_adversaries
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 <= b1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )


###### 2: Best case controller
u2 = cp.Variable( (2,1) )
Q2 = cp.Parameter( (1,2), value = np.zeros((1,2)) )
num_constraints2 = num_robots - 1 + num_adversaries
# minimze A u s.t to other constraints
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 <= b2]
const2 += [cp.abs(u2[0,0])<=40.0]
const2 += [cp.abs(u2[1,0])<=40.0]
objective2 = cp.Minimize( Q2 @ u2 )
best_controller = cp.Problem( objective2, const2 )

##########################################################################################

for i in range(num_steps):
    
    const_index = 0
    
    ## Greedy's nominal movement
    u_greedy_nominal = np.array([1.0, 0.0])
    greedy_nominal[0].step(u_greedy_nominal)
    
    ## Greedy's believed movement
    V_nominal, dV_dx_nominal = greedy[0].lyapunov( greedy_nominal[0].X  )
    u_greedy_nominal = -1.0 * dV_dx_nominal.T /np.linalg.norm( dV_dx_nominal )
    
    ## Greedy actual movement
    V, dV_dx = greedy[0].lyapunov( robots[0].X )
    u_greedy = -5.0 * dV_dx.T / np.linalg.norm( dV_dx )
    greedy[0].step(u_greedy)
    
    # Move nominal agents
    for j in range(num_robots):
        u_nominal = np.array([0,1.0])
        robots_nominal[j].step( u_nominal )
        V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
        robots[j].U_nominal = -1.0*dV_dx.T/np.linalg.norm(dV_dx)
        
    
    for j in range(num_robots):
        
        const_index = 0
                        
        # greedy
        for k in range(num_adversaries):
            h, dh_dxi, dh_dxk = robots[j].agent_barrier(greedy[k], d_min);  
            print(h)
                
            # Control QP constraint
            robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
            robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - dh_dxk @ ( greedy[k].f() + greedy[k].g() @ greedy[k].U ) - robots[j].adv_alpha[k] * h
            const_index = const_index + 1

            # Best Case LP objective
            robots[j].adv_objective[k] = dh_dxi @ robots[j].g()
            
        for k in range(num_robots):
            
            if k==j:
                continue
            
            h, dh_dxi, dh_dxk = robots[j].agent_barrier(robots[k], d_min);
            print(h)
                
            # Control QP constraint
            robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
            robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - dh_dxk @ ( robots[k].f() + robots[k].g() @ robots[k].U ) - robots[j].robot_alpha[k] * h
            const_index = const_index + 1
            
            # Best Case LP objective
            robots[j].robot_objective[k] = dh_dxi @ robots[j].g()
            
        
        
    for j in range(num_robots):
        
        const_index = 0      
        # Constraints in LP and QP are same      
        A1.value = robots[j].A1
        A2.value = robots[j].A1
        b1.value = robots[j].b1
        b2.value = robots[j].b1
        
        # Solve for trust factor
        
        for k in range(num_adversaries):
            Q2 = robots[j].adv_objective[k]
            best_controller.solve()
                        
            h, dh_dxi, dh_dxk = robots[j].agent_barrier(greedy[k], d_min);              
            A = dh_dxk
            b = -robots[j].adv_alpha[0] * h  - dh_dxi @ u2.value #- dh_dxi @ robots[j].U
            
            robots[j].trust_adv = compute_trust( A, b, u_greedy, u_greedy_nominal, h, min_dist, h_min )        
            print(f"{j}'s Trust of {k} adversary: {best_controller.status}: {robots[j].trust_adv} ")    
            # robots[j].adv_alpha[0] = robots[j].adv_alpha[0] + alpha_der_max * robots[j].trust_adv
            if (robots[j].adv_alpha[0]<0):
                robots[j].adv_alpha[0] = 0.01
            
            
        for k in range(num_robots):
            if k==j:
                continue
        
            Q2 = robots[j].robot_objective[k]
            best_controller.solve()
                    
            h, dh_dxi, dh_dxk = robots[j].agent_barrier(robots[k], d_min);
            A = dh_dxk
            b = -robots[j].robot_alpha[k] * h - dh_dxi @ u2.value  #- dh_dxi @ robots[j].U  # need best case U here. not previous U
            
            robots[j].trust_robot = compute_trust( A, b, robots[k].U, robots[k].U_nominal, h, min_dist, h_min )            
            print(f"{j}'s Trust of {k} robot: {best_controller.status}: {robots[j].trust_adv}")
            # robots[j].robot_alpha[k] = robots[j].robot_alpha[k] + alpha_der_max * robots[j].trust_robot
            if (robots[j].robot_alpha[k]<0):
                robots[j].robot_alpha[k] = 0.01
        
        # Solve for control input
        u1_ref.value = robots[j].U_nominal
        cbf_controller.solve()
        print(f"{j}'s input: {cbf_controller.status}")
        robots[j].nextU = u1.value        
        
    for j in range(num_robots):
        robots[j].step( robots[j].nextU )
        robots[j].render_plot()
    
    t = t + dt
    
    fig.canvas.draw()
    fig.canvas.flush_events()