from cProfile import label
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from robot_models.obstacles import *
from utils.utils import *
from graph_utils import *

from matplotlib.animation import FFMpegWriter

plt.rcParams.update({'font.size': 15}) #27

# Sim Parameters                  
dt = 0.05
tf = 9.0 #5.4#8#4.1 #0.2#4.1
num_steps = int(tf/dt)
t = 0
d_min_obstacles = 1.0 #0.1
d_min_agents = 0.2 #0.4
d_max = 2.0

h_min = 1.0##0.4   # more than this and do not decrease alpha
min_dist = 1.0 # 0.1#0.05  # less than this and dercrease alpha
cbf_extra_bad = 0.0
update_param = True
bigNaN = 10000000


alpha_cbf = 0.7 #0.8
alpha_der_max = 0.1 #0.5#1.0#0.5

# Plot                  
plt.ion()
fig = plt.figure()
# ax = plt.axes(xlim=(0,7),ylim=(-0.5,8)) 
ax = plt.axes(xlim=(0,7),ylim=(-0.5,10)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_aspect(1)


num_adversaries = 0
num_obstacles = 3
num_connectivity = 1
alpha = 0.1

save_plot = False
movie_name = 'test.mp4'

# agents
robots = []
num_robots = 6
# robots.append( Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
# robots.append( Unicycle(np.array([2.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
# robots.append( Unicycle(np.array([3.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )

robots.append( SingleIntegrator2D(np.array([3,1.5]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([2.6,0]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([2.5,0.8]), dt, ax, num_robots=num_robots, id = 3, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([3.0,0.8]), dt, ax, num_robots=num_robots, id = 4, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([3.5,0.8]), dt, ax, num_robots=num_robots, id = 5, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )

# agent nominal version
robots_nominal = []

robots_nominal.append( SingleIntegrator2D(np.array([3,1.5]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=alpha) )
robots_nominal.append( SingleIntegrator2D(np.array([2.9,0]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([2.5,0.8]), dt, ax, num_robots=num_robots, id = 3, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([3.0,0.8]), dt, ax, num_robots=num_robots, id = 4, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([3.5,0.8]), dt, ax, num_robots=num_robots, id = 5, color='g',palpha=alpha ) )
U_nominal = np.zeros((2,num_robots))

obstacles = []
obstacles.append( circle( 1.8,2.5,1.0,ax,0 ) ) # x,y,radius, ax, id
obstacles.append( circle( 4.2,2.5,1.0,ax,1 ) )
obstacles.append( circle( 6.2,2.5,1.0,ax,2 ) )

# plt.ioff()
# plt.show()

# exit()
# Adversarial agents
# adversary = []
# adversary.append( SingleIntegrator2D(np.array([0,4]), dt, ax) )


############################## Optimization problems ######################################

###### 1: CBF Controller
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1 + num_adversaries + num_obstacles + num_connectivity
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
slack_constraints1 = cp.Parameter( (num_constraints1,1), value = np.zeros((num_constraints1,1)) )
const1 = [A1 @ u1 <= b1 + slack_constraints1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )


###### 2: Best case controller
u2 = cp.Variable( (2,1) )
Q2 = cp.Parameter( (1,2), value = np.zeros((1,2)) )
num_constraints2 = num_robots - 1 + num_adversaries + num_obstacles + num_connectivity
# minimze A u s.t to other constraints
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
slack_constraints2 = cp.Parameter( (num_constraints2,1), value = np.zeros((num_constraints1,1)) )
const2 = [A2 @ u2 <= b2 + slack_constraints2]
const2 += [ cp.abs( u2[0,0] ) <= 7.0 ]
const2 += [ cp.abs( u2[1,0] ) <= 7.0 ]
objective2 = cp.Minimize( Q2 @ u2 )
best_controller = cp.Problem( objective2, const2 )
       
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

tp = []

with writer.saving(fig, movie_name, 100): 

    for i in range(num_steps):
        
        # Laplacina for connectivity
        L = weighted_connectivity_undirected_laplacian(robots, max_dist = 4.5)
        Lambda, V = laplacian_eigen( L )
        print(f" Eigen value:{ Lambda[1] } ")
        lambda2_dx( robots, L, Lambda[1], V[:,1].reshape(-1,1) )
        
        const_index = 0
            
        # Move nominal agents
        for j in range(num_robots):
            # u_nominal = np.array([1.0,0.0])
            u_nominal = np.array([0.0,1.0])
            robots_nominal[j].step( u_nominal )
            V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
            robots[j].x_dot_nominal = -1.0*dV_dx.T/np.linalg.norm(dV_dx) # 3.0
            robots[j].U_ref = robots[j].nominal_input( robots_nominal[j] )
            robots_nominal[j].render_plot()
        
        #  Get inequality constraints
        for j in range(num_robots):
            
            const_index = 0
                
            # obstacles
            for k in range(num_obstacles):
                h, dh_dxi, dh_dxk = robots[j].agent_barrier(obstacles[k], d_min_obstacles);  
                robots[j].obs_h[0,k] = h
                
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - robots[j].obs_alpha[0,k] * h
            
                # Best Case LP objective
                robots[j].obs_objective[k] = dh_dxi @ robots[j].g()                
                
                const_index = const_index + 1
                
            # Min distance constraint
            for k in range(num_robots):
                
                print(f" j:{j}, k:{k}, dist:{ np.linalg.norm( robots[j].X[0:2] - robots[k].X[0:2] ) } ")
                
                if k==j:
                    continue
                
                # if j==2:
                #     if k==1:
                #         robots[j].slack_constraint[const_index,:] = bigNaN
                
                h, dh_dxj, dh_dxk = robots[j].agent_barrier(robots[k], d_min_agents)
                robots[j].robot_h[0,k] = h
                if h < 0:
                    robots[j].slack_constraint[const_index,0] = 0.0
                    
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxj @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxj @ robots[j].f() - dh_dxk @ ( robots[k].f() + robots[k].g() @ robots[k].U ) - cbf_extra_bad - robots[j].robot_alpha[0,k] * h
                
                # Best Case LP objective
                robots[j].robot_objective[k] = dh_dxj @ robots[j].g()
                
                const_index = const_index + 1
                
            # Max distance constraint for connectivity            
            if j!=0:                
                h, dh_dxj, dh_dxk = robots[j].connectivity_barrier(robots[0], d_max)
                robots[j].robot_connectivity_h = h
                if h < 0:
                    robots[j].slack_constraint[const_index,0] = 0.0
                
                    
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxj @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxj @ robots[j].f() - dh_dxk @ ( robots[0].f() + robots[0].g() @ robots[0].U ) - cbf_extra_bad - robots[j].robot_connectivity_alpha[0,0] * h
                
                # Best Case LP objective
                robots[j].robot_connectivity_objective = dh_dxj @ robots[j].g()
                
                const_index = const_index + 1
                
            
            
        # Design control input and update alphas with trust
        for j in range(num_robots):
            
            const_index = 0      
            # Constraints in LP and QP are same      
            A1.value = robots[j].A1
            A2.value = robots[j].A1
            b1.value = robots[j].b1
            b2.value = robots[j].b1
            slack_constraints1.value = robots[j].slack_constraint
            slack_constraints2.value = robots[j].slack_constraint
            
            # Solve for trust factor            
            if update_param:
    
                # Obstacle Trust
                # for k in range(num_obstacles):
                #     Q2.value = robots[j].obs_objective[k]
                #     best_controller.solve(solver=cp.GUROBI)
                #     if best_controller.status!='optimal':
                #         print(f"LP status:{best_controller.status}")
                                
                #     h, dh_dxj, dh_dxk = robots[j].agent_barrier(obstacles[k], d_min_obstacles)  
                    
                #     # print(f"out, h:{h}, j:{j}, k:{k}, alpha:{robots[j].adv_alpha[0]}")
                #     # assert(h<0.03)           
                #     A = np.array([[0.0,0.0]])
                #     b = -robots[j].obs_alpha[0,k] * h  - dh_dxj @ ( robots[j].f() + robots[j].g() @ u2.value )   #- dh_dxi @ robots[j].U
                    
                #     robots[j].trust_obs[0,k] = compute_trust( A, b, np.array([0,0]).reshape(-1,1), np.array([0,0]).reshape(-1,1), h, min_dist, h_min )  
                #     # if robots[j].trust_obs[0,k]<0:
                #     # print(f"{j}'s Trust of {k} Obstacle: {best_controller.status}: {robots[j].trust_obs[0,k]}, h:{h} ")    
                #     robots[j].obs_alpha[0,k] = robots[j].obs_alpha[0,k] + alpha_der_max * robots[j].trust_obs[0,k]
                #     if robots[j].obs_alpha[0,k] > 0.8:
                #         robots[j].obs_alpha[0,k] = 0.8
                #     if (robots[j].obs_alpha[0,k]<0):
                #         robots[j].obs_alpha[0,k] = 0.01
                    
                # Min distance
                const_index = num_obstacles
                # print(f"j:{j}, const:{const_index}")
                for k in range(num_robots):
                    if k==j:
                        continue
                    
                    if robots[j].slack_constraint[const_index,0] > bigNaN * 0.99:
                        # print(f"j:{j} part, const:{const_index}")
                        const_index += 1
                        continue
                
                    Q2.value = robots[j].robot_objective[k]
                    best_controller.solve(solver=cp.GUROBI, reoptimize=True)#, verbose=True)
                    if best_controller.status!='optimal':
                        print(f"robot: {j}, Min distance LP status:{best_controller.status}")
                        robots[j].slack_constraint[-1,0] = bigNaN
                        slack_constraints1.value = robots[j].slack_constraint # robots[j].slack_constraint
                        slack_constraints2.value = robots[j].slack_constraint #  robots[j].slack_constraint
                        best_controller.solve(solver=cp.GUROBI, reoptimize=True)
                        if best_controller.status!='optimal':
                            for kk in range(num_constraints2-1):
                                if b2.value[kk,0] < 0:
                                    robots[j].slack_constraint[kk,0] = bigNaN
                            slack_constraints1.value = robots[j].slack_constraint # robots[j].slack_constraint
                            slack_constraints2.value = robots[j].slack_constraint #  robots[j].slack_constraint
                            best_controller.solve(solver=cp.GUROBI, reoptimize=True)
                            if best_controller.status!='optimal':
                                print(f"serious ERROR")
                                exit()
                    
                            
                    h, dh_dxi, dh_dxk = robots[j].agent_barrier(robots[k], d_min_agents);
                    if  robots[j].slack_constraint[const_index,0] > bigNaN * 0.99:   
                        const_index += 1  
                        # print(f"j:{j} part, const:{const_index}")
                        continue
                    assert(h<0.01)
                    A = dh_dxk 
                    b = -robots[j].robot_alpha[0,k] * h - dh_dxi @ ( robots[j].f() + robots[j].g() @  u2.value) #- dh_dxi @ robots[j].U  # need best case U here. not previous U
                    
                    robots[j].trust_robot[0,k] = compute_trust( A, b, robots[k].f() + robots[k].g() @ robots[k].U, robots[k].x_dot_nominal, h, min_dist, h_min )            
                    # if robots[j].trust_robot[0,k]<0:
                    #     print(f"{j}'s Trust of {k} robot: {best_controller.status}: {robots[j].trust_robot[0,k]}, h:{h}")
                    robots[j].robot_alpha[0,k] = robots[j].robot_alpha[0,k] + alpha_der_max * robots[j].trust_robot[0,k]
                    if (robots[j].robot_alpha[0,k]<0):
                        robots[j].robot_alpha[0,k] = 0.01
                        
                    const_index += 1
                    # print(f"j:{j} full, const:{const_index}")
                        
                # Max distance
                if j!=0 and robots[j].slack_constraint[-1,0] < bigNaN * 0.99:
                    Q2.value = robots[j].robot_connectivity_objective
                    best_controller.solve(solver=cp.GUROBI, reoptimize=True)#, verbose=True)
                    if best_controller.status!='optimal':
                        print(f"robot: {j}, Max distance LP status:{best_controller.status}")
                        robots[j].slack_constraint[-1,0] = bigNaN
                        slack_constraints1.value = robots[j].slack_constraint # robots[j].slack_constraint
                        slack_constraints2.value = robots[j].slack_constraint #  robots[j].slack_constraint
                        best_controller.solve(solver=cp.GUROBI, reoptimize=True)
                        if best_controller.status!='optimal':
                            for kk in range(num_constraints2-1):
                                if b2.value[kk,0] < 0:
                                    robots[j].slack_constraint[kk,0] = bigNaN
                            slack_constraints1.value = robots[j].slack_constraint # robots[j].slack_constraint
                            slack_constraints2.value = robots[j].slack_constraint #  robots[j].slack_constraint
                            best_controller.solve(solver=cp.GUROBI, reoptimize=True)
                            if best_controller.status!='optimal':
                                print(f"serious ERROR")
                                exit()
                    
                    if  robots[j].slack_constraint[-1,0] > bigNaN * 0.99:     
                        continue
                    h, dh_dxi, dh_dxk = robots[j].connectivity_barrier(robots[0], d_max);
                    
                    assert(h<0.01)
                    A = dh_dxk 
                    b = -robots[j].robot_connectivity_alpha[0,0] * h - dh_dxi @ ( robots[j].f() + robots[j].g() @  u2.value) #- dh_dxi @ robots[j].U  # need best case U here. not previous U
                    
                    robots[j].trust_robot_connectivity = compute_trust( A, b, robots[0].f() + robots[0].g() @ robots[0].U, robots[0].x_dot_nominal, h, min_dist, h_min )            
                    # if robots[j].trust_robot[0,k]<0:
                    #     print(f"{j}'s Trust of {k} robot: {best_controller.status}: {robots[j].trust_robot[0,k]}, h:{h}")
                    robots[j].robot_connectivity_alpha[0,0] = robots[j].robot_connectivity_alpha[0,0] + alpha_der_max * robots[j].trust_robot_connectivity
                    if (robots[j].robot_connectivity_alpha[0,0]<0):
                        robots[j].robot_connectivity_alpha[0,0] = 0.01
                        
            # print(f" robot :{j}, alphas connectivity:{ robots[j].robot_connectivity_alpha }, robots:{ robots[j].robot_alpha }, obs:{ robots[j].obs_alpha }, h:{ robots[j].robot_connectivity_h } ")
                
            # Plotting
            robots[j].adv_alphas = np.append( robots[j].adv_alphas, robots[j].adv_alpha, axis=0 )
            robots[j].trust_advs = np.append( robots[j].trust_advs, robots[j].trust_adv, axis=0 )
            robots[j].robot_alphas = np.append( robots[j].robot_alphas, robots[j].robot_alpha, axis=0 )
            robots[j].trust_robots = np.append( robots[j].trust_robots, robots[j].trust_robot, axis=0 )
            robots[j].robot_hs = np.append( robots[j].robot_hs, robots[j].robot_h, axis=0 )
            robots[j].adv_hs = np.append( robots[j].adv_hs, robots[j].adv_h, axis=0 )
            robots[j].obs_alphas = np.append( robots[j].obs_alphas, robots[j].obs_alpha, axis=0 )
            robots[j].trust_obss = np.append( robots[j].trust_obss, robots[j].trust_obs, axis=0 )
            robots[j].obs_hs = np.append( robots[j].obs_hs, robots[j].obs_h, axis=0 )
            robots[j].robot_connectivity_alphas = np.append( robots[j].robot_connectivity_alphas, robots[j].robot_connectivity_alpha, axis = 0 )
            robots[j].robot_connectivity_hs = np.append( robots[j].robot_connectivity_hs, robots[j].robot_connectivity_h, axis=0 )
            
            # Solve for control input
            if j==0:
                u1_ref.value = robots[j].U_ref
            else:
                u1_ref.value = 2*robots[j].lambda2_dx.reshape(-1,1)
            cbf_controller.solve(solver=cp.GUROBI)#, verbose=True)
            if cbf_controller.status!='optimal':
                print(f"{j}'s input: {cbf_controller.status}")
                exit()
            robots[j].nextU = u1.value       

         
        for j in range(num_robots):
            robots[j].step( robots[j].nextU )
            robots[j].render_plot()
            # print(f"{j} state: {robots[j].X[1,0]}, input:{robots[j].nextU[0,0]}, {robots[j].nextU[1,0]}")
        
        t = t + dt
        tp.append(t)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        writer.grab_frame()
    
plt.ioff()   

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# matplotlib.rc('font', **font)


# Plot

####### Alphas #######
# Robot 0
figure1, axis1 = plt.subplots(2, 2)
if num_adversaries>0:
    axis1[0,0].plot(tp,robots[0].adv_alphas[1:,0],'r',label='Adversary')
if num_obstacles>0:
    axis1[0,0].plot(tp,robots[0].obs_alphas[1:,0],'r',label='Obstacles')
axis1[0,0].plot(tp,robots[0].robot_alphas[1:,1],'g',label='Robot 2')
axis1[0,0].plot(tp,robots[0].robot_alphas[1:,2],'k',label='Robot 3')
axis1[0,0].set_title('Robot 1 alphas')
axis1[0,0].set_xlabel('time (s)')
axis1[0,0].legend()

# Robot 1
if num_adversaries>0:
    axis1[1,0].plot(tp,robots[1].adv_alphas[1:,0],'r',label='Adversary')
if num_obstacles>0:
    axis1[1,0].plot(tp,robots[1].obs_alphas[1:,0],'r',label='Obstacles')
axis1[1,0].plot(tp,robots[1].robot_alphas[1:,0],'g',label='Robot 1')
axis1[1,0].plot(tp,robots[1].robot_alphas[1:,2],'k',label='Robot 3')
axis1[1,0].set_title('Robot 2 alphas')
axis1[1,0].set_xlabel('time (s)')
axis1[1,0].legend()

# Robot 2
if num_adversaries>0:
    axis1[0,1].plot(tp,robots[2].adv_alphas[1:,0],'r',label='Adversary')
if num_obstacles>0:
    axis1[0,1].plot(tp,robots[2].obs_alphas[1:,0],'r',label='Obstacles')
axis1[0,1].plot(tp,robots[2].robot_alphas[1:,0],'g',label='Robot 1')
axis1[0,1].plot(tp,robots[2].robot_alphas[1:,1],'k',label='Robot 2')
axis1[0,1].set_title('Robot 3 alphas')
axis1[0,1].set_xlabel('time (s)')
axis1[0,1].legend()

#### TRUST ######
# Robot 0
figure2, axis2 = plt.subplots(2, 2)
if num_adversaries>0:
    axis2[0,0].plot(tp,robots[0].trust_advs[1:,0],'r',label='Adversary')
if num_obstacles>0:
    axis2[0,0].plot(tp,robots[0].trust_obss[1:,0],'r',label='Obstacles')
axis2[0,0].plot(tp,robots[0].trust_robots[1:,1],'g',label='Robot 2')
axis2[0,0].plot(tp,robots[0].trust_robots[1:,2],'k',label='Robot 3')
axis2[0,0].set_title('Robot 1 trust')
axis2[0,0].set_xlabel('time (s)')
axis2[0,0].legend()

# Robot 1
if num_adversaries>0:
    axis2[1,0].plot(tp,robots[1].trust_advs[1:,0],'r',label='Adversary')
if num_obstacles>0:
    axis2[1,0].plot(tp,robots[1].trust_obss[1:,0],'r',label='Obstacles')
axis2[1,0].plot(tp,robots[1].trust_robots[1:,0],'g',label='Robot 1')
axis2[1,0].plot(tp,robots[1].trust_robots[1:,2],'k',label='Robot 3')
axis2[1,0].set_title('Robot 2 trust')
axis2[1,0].set_xlabel('time (s)')
axis2[1,0].legend()

# Robot 2
if num_adversaries>0:
    axis2[0,1].plot(tp,robots[2].trust_advs[1:,0],'r',label='Adversary')
if num_obstacles>0:
    axis2[0,1].plot(tp,robots[2].trust_obss[1:,0],'r',label='Obstacles')
axis2[0,1].plot(tp,robots[2].trust_robots[1:,0],'g',label='Robot 1')
axis2[0,1].plot(tp,robots[2].trust_robots[1:,1],'k',label='Robot 2')
axis2[0,1].set_title('Robot 3 trust')
axis2[0,1].set_xlabel('time (s)')
axis2[0,1].legend()

plt.show()
exit()

figure9, axis9 = plt.subplots(1, 2)
if num_adversaries>0:
    axis9[0].plot(tp,robots[0].adv_alphas[1:,0],'r',label='Adversary')
axis9[0].plot(tp,robots[0].robot_alphas[1:,1],'g',label='Robot 2')
axis9[0].plot(tp,robots[0].robot_alphas[1:,2],'k',label='Robot 3')
axis9[0].set_title(r'Robot 1 $\alpha$')
axis9[0].set_xlabel('time (s)')
axis9[0].legend()

if num_adversaries>0:
    axis9[1].plot(tp,robots[0].adv_hs[1:,0],'r',label='Adversary - Proposed')
axis9[1].plot(tp,robots[0].robot_hs[1:,1],'g',label='Robot 2 - Proposed')
axis9[1].plot(tp,robots[0].robot_hs[1:,2],'k',label='Robot 3 - Proposed')
if num_adversaries>0:
    axis9[1].plot(tp,robots_default[0].adv_hs[1:,0],'r--',label=r'Adversary - fixed $\alpha$')
axis9[1].plot(tp,robots_default[0].robot_hs[1:,1],'g--',label=r'Robot - fixed $\alpha$')
axis9[1].plot(tp,robots_default[0].robot_hs[1:,2],'k--',label=r'Robot - fixed $\alpha$')
axis9[1].set_title('Robot 1 CBFs')
axis9[1].set_xlabel('time (s)')
axis9[1].legend()

figure6, axis6 = plt.subplots(1, 1)
if num_adversaries>0:
    axis6.plot(tp,robots[0].trust_advs[1:,0]/2,'r',label='Robot 1 trust of Adversary')
    axis6.plot(tp,robots[1].trust_advs[1:,0]/2,'r--',label='Robot 2 trust of Adversary')
axis6.plot(tp,robots[0].trust_robots[1:,1]/2,'g',label='Robot 1 trust of 2 ')
axis6.plot(tp,robots[1].trust_robots[1:,0]/2,'g--',label='Robot 2 trust of 1 ')
axis6.plot(tp,robots[0].trust_robots[1:,2]/2,'k',label='Robot 1 trust of 3 ')
axis6.plot(tp,robots[2].trust_robots[1:,0]/2,'k--',label='Robot 3 trust of 1 ')
axis6.legend()
axis6.set_xlabel('time (s)')

######### Barriers ##########
# Robot 0
figure3, axis3 = plt.subplots(2, 2)
axis3[0,0].plot(tp,robots[0].adv_hs[1:,0],'r',label='Adversary')
axis3[0,0].plot(tp,robots[0].robot_hs[1:,1],'g',label='Robot 2')
axis3[0,0].plot(tp,robots[0].robot_hs[1:,2],'k',label='Robot 3')
axis3[0,0].set_title('Robot 1 CBFs')
axis3[0,0].set_xlabel('time (s)')
axis3[0,0].legend()

# Robot 1
axis3[1,0].plot(tp,robots[1].adv_hs[1:,0],'r',label='Adversary')
axis3[1,0].plot(tp,robots[1].robot_hs[1:,0],'g',label='Robot 1')
axis3[1,0].plot(tp,robots[1].robot_hs[1:,2],'k',label='Robot 3')
axis3[1,0].set_title('Robot 2 CBFs')
axis3[1,0].set_xlabel('time (s)')
axis3[1,0].legend()

# Robot 2
axis3[0,1].plot(tp,robots[2].adv_hs[1:,0],'r',label='Adversary')
axis3[0,1].plot(tp,robots[2].robot_hs[1:,0],'g',label='Robot 1')
axis3[0,1].plot(tp,robots[2].robot_hs[1:,1],'k',label='Robot 3')
axis3[0,1].set_title('Robot 3 CBFs')
axis3[0,1].set_xlabel('time (s)')
axis3[0,1].legend()

figure5, axis5 = plt.subplots(1, 1)
axis5.plot(tp,-robots[0].adv_hs[1:,0],'r',label='Adversary - Proposed')
axis5.plot(tp,-robots[0].robot_hs[1:,1],'g',label='Robot 2 - Proposed')
axis5.plot(tp,-robots[0].robot_hs[1:,2],'k',label='Robot 3 - Proposed')
axis5.plot(tp,-robots_default[0].adv_hs[1:,0],'r--',label=r'Adversary - fixed $\alpha$')
axis5.plot(tp,-robots_default[0].robot_hs[1:,1],'g--',label=r'Robot - fixed $\alpha$')
axis5.plot(tp,-robots_default[0].robot_hs[1:,2],'k--',label=r'Robot - fixed $\alpha$')
axis5.legend()
axis5.set_xlabel('time (s)')
# plt.show()

figure8, axis8 = plt.subplots(1, 1)
axis8.plot(tp,robots[0].adv_alphas[1:,0],'r',label='Adversary')
axis8.plot(tp,robots[0].robot_alphas[1:,1],'g',label='Robot 2')
axis8.plot(tp,robots[0].robot_alphas[1:,2],'k',label='Robot 3')
axis8.set_title('Robot 1 alphas')
axis8.set_xlabel('time (s)')
axis8.legend()

figure4, axis4 = plt.subplots(1, 1)
plt.rcParams.update({'font.size': 10})
plt.xlim([0,7])
plt.ylim([-0.5,8])
# axis3 = plt.axes(xlim=(0,8),ylim=(-0.5,8))
ax.set_xlabel("X")
ax.set_ylabel("Y")





tp = -1 + 2*np.asarray(tp)/tp[-1]
tp = tp/np.max(tp)


# Reduce Data


cc = np.tan(np.asarray(tp))
im1 = axis4.scatter( robots[0].Xs[0,1:], robots[0].Xs[1,1:],c=cc )
axis4.scatter( robots[1].Xs[0,1:], robots[1].Xs[1,1:],c=cc )
axis4.scatter( robots[2].Xs[0,1:], robots[2].Xs[1,1:],c=cc )

im2 = axis4.scatter( greedy[0].Xs[0,1:], greedy[0].Xs[1,1:],c=cc, cmap = 'CMRmap' )
axis4.scatter( greedy[1].Xs[0,1:], greedy[1].Xs[1,1:],c=cc, cmap = 'CMRmap' )
# axis4.scatter( greedy[2].Xs[0,1:], greedy[2].Xs[1,1:],c=cc, cmap = 'CMRmap' )

axis4.scatter( robots_default[0].Xs[0,1:], robots_default[0].Xs[1,1:],c=cc, alpha = 0.1 )
axis4.scatter( robots_default[1].Xs[0,1:], robots_default[1].Xs[1,1:],c=cc, alpha = 0.1 )
axis4.scatter( robots_default[2].Xs[0,1:], robots_default[2].Xs[1,1:],c=cc, alpha = 0.1 )
axis4.scatter( greedy_default[0].Xs[0,1:], greedy_default[0].Xs[1,1:],c=cc, cmap = 'CMRmap',alpha=0.1 )

figure4.colorbar(im1, ax=axis4)
figure4.colorbar(im2, ax=axis4)

# plt.colorbar(sc)
#######################################


figure7, axis7 = plt.subplots(1, 1)

plt.xlim([0,7])
plt.ylim([-0.5,8])
# axis3 = plt.axes(xlim=(0,8),ylim=(-0.5,8))
axis7.set_xlabel("X")
axis7.set_ylabel("Y")


tp = -1 + 2*np.asarray(tp)/tp[-1]
tp = tp/np.max(tp)
# r0 = np.zeros()

# tps = []
# for idx, value in enumerate(tp):
#     if idx % 4==0:
#         tps.append(value)


div = 10
cc = np.tan(np.asarray(tp[1::div]))
im1 = axis7.scatter( robots[0].Xs[0,1::div], robots[0].Xs[1,1::div],c=cc,rasterized=True )
axis7.scatter( robots[1].Xs[0,1::div], robots[1].Xs[1,1::div],c=cc )
axis7.scatter( robots[2].Xs[0,1::div], robots[2].Xs[1,1::div],c=cc )

im2 = axis7.scatter( greedy[0].Xs[0,1::div], greedy[0].Xs[1,1::div],c=cc, cmap = 'CMRmap' )
axis7.scatter( greedy[1].Xs[0,1::div], greedy[1].Xs[1,1::div],c=cc, cmap = 'CMRmap' )
axis7.scatter( greedy[2].Xs[0,1::div], greedy[2].Xs[1,1::div],c=cc, cmap = 'CMRmap' )

axis7.scatter( robots_default[0].Xs[0,1::div], robots_default[0].Xs[1,1::div],c=cc, alpha = 0.1 )
axis7.scatter( robots_default[1].Xs[0,1::div], robots_default[1].Xs[1,1::div],c=cc, alpha = 0.1 )
axis7.scatter( robots_default[2].Xs[0,1::div], robots_default[2].Xs[1,1::div],c=cc, alpha = 0.1 )
axis7.scatter( greedy_default[0].Xs[0,1::div], greedy[0].Xs[1,1::div],c=cc, cmap = 'CMRmap',alpha=0.1 )

figure7.colorbar(im1, ax=axis7)
figure7.colorbar(im2, ax=axis7)
# plt.colorbar(sc)





###########################


# figure1.savefig("alphas.eps")
# figure2.savefig("trust.eps")
# figure3.savefig("cbfs.eps")
# figure4.savefig("trajectory.eps")

# axis4.legend()
axis4.set_xlabel('X')
axis4.set_ylabel('Y')

plt.show()



print("hello")

#'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

if save_plot:
    figure1.savefig("alphas.eps")
    figure2.savefig("trust.eps")
    figure3.savefig("cbfs.eps")
    
    axis4.set_rasterization_zorder(1)
    figure4.savefig("trajectory.eps", dpi=50, rasterized=True)
    figure4.savefig("trajectory.png")
    figure5.savefig("barriers_robot_1.eps")
    figure6.savefig("trusts.eps")
    figure8.savefig("alphas.eps")
    figure8.savefig("alphas.png")