from cProfile import label
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from utils.utils import *

from matplotlib.animation import FFMpegWriter

plt.rcParams.update({'font.size': 27})

# Sim Parameters                  
dt = 0.05
tf = 5.4#8#4.1 #0.2#4.1
num_steps = int(tf/dt)
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
movie_name = 'nominal_adapt.mp4'

# agents
robots = []
num_robots = 3
robots.append( Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries ) )
robots.append( Unicycle(np.array([2.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries ) )
robots.append( Unicycle(np.array([3.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries ) )

robots_default = []
robots_default.append( Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, plot=default_plot ) )
robots_default.append( Unicycle(np.array([2.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, plot=default_plot ) )
robots_default.append( Unicycle(np.array([3.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, plot=default_plot ) )

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

greedy_default = []
greedy_default.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=1.0, plot=default_plot) )
greedy_default.append( SingleIntegrator2D(np.array([0,5]), dt, ax, color='r',palpha=1.0, plot=default_plot) )
greedy_default.append( SingleIntegrator2D(np.array([7,7]), dt, ax, color='r',palpha=1.0, plot=default_plot) )

greedy_nominal = []
greedy_nominal.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=alpha) )
greedy_nominal.append( SingleIntegrator2D(np.array([0,5]), dt, ax, color='r',palpha=alpha) )
greedy_nominal.append( SingleIntegrator2D(np.array([7,7]), dt, ax, color='r',palpha=alpha) )


# Adversarial agents
# adversary = []
# adversary.append( SingleIntegrator2D(np.array([0,4]), dt, ax) )


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
# const2 += [cp.abs(u2[0,0])<=10.0]
# const2 += [cp.abs(u2[1,0])<=40.0]
objective2 = cp.Minimize( Q2 @ u2 )
best_controller = cp.Problem( objective2, const2 )

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

       
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

tp = []

with writer.saving(fig, movie_name, 100): 

    for i in range(num_steps):
        
        const_index = 0
        
        
        
        # Greedy agents
        for j in range(num_adversaries):
            ## Greedy's nominal movement
            if j==0 or j==1:
                u_greedy_nominal = np.array([1.0, 0.0])
            else:
                u_greedy_nominal = np.array([-1.0,0.0])
            greedy_nominal[j].step(u_greedy_nominal)
            
            ## Greedy's believed movement
            V_nominal, dV_dx_nominal = greedy[j].lyapunov( greedy_nominal[j].X  )
            greedy[j].x_dot_nominal = -1.0 * dV_dx_nominal.T /np.linalg.norm( dV_dx_nominal )
            
            V_nominal, dV_dx_nominal = greedy_default[j].lyapunov( greedy_nominal[j].X  )
            greedy_default[j].x_dot_nominal = -1.0 * dV_dx_nominal.T /np.linalg.norm( dV_dx_nominal )
            
            ## Greedy actual movement
            if j==0:
                V, dV_dx = greedy[0].lyapunov( robots[0].X )
                greedy[j].U_ref = -1.0 * dV_dx.T / np.linalg.norm( dV_dx )
                
                V, dV_dx = greedy_default[0].lyapunov( robots_default[0].X )
                greedy_default[j].U_ref = -1.0 * dV_dx.T / np.linalg.norm( dV_dx )
            else:
                greedy[j].U_ref = u_greedy_nominal
                
                greedy_default[j].U_ref = u_greedy_nominal
            
        # Move nominal agents
        for j in range(num_robots):
            u_nominal = np.array([1.0,0.0])
            robots_nominal[j].step( u_nominal )
            V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
            robots[j].x_dot_nominal = -3.0*dV_dx.T/np.linalg.norm(dV_dx)
            robots[j].U_ref = robots[j].nominal_input( robots_nominal[j] )
            robots_nominal[j].render_plot()
            
            V, dV_dx = robots_default[j].lyapunov(robots_nominal[j].X)
            robots_default[j].x_dot_nominal = -3.0*dV_dx.T/np.linalg.norm(dV_dx)
            robots_default[j].U_ref = robots_default[j].nominal_input( robots_nominal[j] )
        
        for j in range(num_robots):
            
            const_index = 0
                            
            # greedy
            for k in range(num_adversaries):
                h, dh_dxi, dh_dxk = robots[j].agent_barrier(greedy[k], d_min);  
                robots[j].adv_h[0,k] = h
                
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - dh_dxk @ ( greedy[k].f() + greedy[k].g() @ greedy[k].U ) - cbf_extra_bad - robots[j].adv_alpha[0,k] * h
                

                # Best Case LP objective
                robots[j].adv_objective[k] = dh_dxi @ robots[j].g()
                
                ### Default################################
                h, dh_dxi, dh_dxk = robots_default[j].agent_barrier(greedy_default[k], d_min);  
                robots_default[j].adv_h[0,k] = h
                
                # Control QP constraint
                robots_default[j].A1[const_index,:] = dh_dxi @ robots_default[j].g()
                robots_default[j].b1[const_index] = -dh_dxi @ robots_default[j].f() - dh_dxk @ ( greedy_default[k].f() + greedy_default[k].g() @ greedy_default[k].U ) - cbf_extra_bad - robots_default[j].adv_alpha[0,k] * h

                # Best Case LP objective
                robots_default[j].adv_objective[k] = dh_dxi @ robots_default[j].g()
                ##########################################
                
                
                const_index = const_index + 1
                
            for k in range(num_robots):
                
                if k==j:
                    continue
                
                h, dh_dxj, dh_dxk = robots[j].agent_barrier(robots[k], d_min);
                robots[j].robot_h[0,k] = h
                    
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxj @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxj @ robots[j].f() - dh_dxk @ ( robots[k].f() + robots[k].g() @ robots[k].U ) - cbf_extra_bad - robots[j].robot_alpha[0,k] * h
                
                # Best Case LP objective
                robots[j].robot_objective[k] = dh_dxj @ robots[j].g()
                
                ### Default ###############################
                h, dh_dxj, dh_dxk = robots_default[j].agent_barrier(robots_default[k], d_min);
                robots_default[j].robot_h[0,k] = h
                    
                # Control QP constraint
                robots_default[j].A1[const_index,:] = dh_dxj @ robots_default[j].g()
                robots_default[j].b1[const_index] = -dh_dxj @ robots_default[j].f() - dh_dxk @ ( robots_default[k].f() + robots_default[k].g() @ robots_default[k].U ) - cbf_extra_bad - robots_default[j].robot_alpha[0,k] * h
                
                # Best Case LP objective
                robots_default[j].robot_objective[k] = dh_dxj @ robots_default[j].g()
                ###########################################
                
                const_index = const_index + 1
                  
        for j in range(num_robots):
            
            const_index = 0      
            # Constraints in LP and QP are same      
            A1.value = robots[j].A1
            A2.value = robots[j].A1
            b1.value = robots[j].b1
            b2.value = robots[j].b1
            
            # Solve for trust factor
            
            if update_param:
                for k in range(num_adversaries):
                    Q2 = robots[j].adv_objective[k]
                    best_controller.solve(solver=cp.GUROBI)
                    if best_controller.status!='optimal':
                        print(f"LP status:{best_controller.status}")
                                
                    h, dh_dxj, dh_dxk = robots[j].agent_barrier(greedy[k], d_min)  
                    
                    # print(f"out, h:{h}, j:{j}, k:{k}, alpha:{robots[j].adv_alpha[0]}")
                    # assert(h<0.03)           
                    A = dh_dxk @ greedy[k].g()
                    b = -robots[j].adv_alpha[0,k] * h  - dh_dxj @ ( robots[j].f() + robots[j].g() @ u2.value ) - dh_dxk @ greedy[k].f() #- dh_dxi @ robots[j].U
                    
                    ## Update u_greedy_nominal based on past observations
                    try:
                        u_greedy_nominal_estimated = greedy[k].Xs[:,-1] - greedy[k].Xs[:,-2]
                    except: 
                        print("Didn't have enough data to estimate nominal")
                        u_greedy_nominal_estimated = u_greedy_nominal
                     
                    #get past observations: previous velocity vector
                    robots[j].trust_adv[0,k] = compute_trust( A, b, greedy[k].f() + greedy[k].g() @ greedy[k].U, u_greedy_nominal_estimated, h, min_dist, h_min )  
                    # robots[j].trust_adv[0,k] = compute_trust( A, b, greedy[k].f() + greedy[k].g() @ greedy[k].U, u_greedy_nominal, h, min_dist, h_min )  
                    # if robots[j].trust_adv[0,k]<0:
                    #     print(f"{j}'s Trust of {k} adversary: {best_controller.status}: {robots[j].trust_adv[0,k]}, h:{h} ")    
                    robots[j].adv_alpha[0,k] = robots[j].adv_alpha[0,k] + alpha_der_max * robots[j].trust_adv[0,k]
                    if (robots[j].adv_alpha[0,k]<0):
                        robots[j].adv_alpha[0,k] = 0.01
                    
                    
                for k in range(num_robots):
                    if k==j:
                        continue
                
                    Q2 = robots[j].robot_objective[k]
                    best_controller.solve()
                    if best_controller.status!='optimal':
                        print(f"LP status:{best_controller.status}")
                            
                    h, dh_dxi, dh_dxk = robots[j].agent_barrier(robots[k], d_min);
                    
                    assert(h<0.01)
                    A = dh_dxk 
                    b = -robots[j].robot_alpha[0,k] * h - dh_dxi @ ( robots[j].f() + robots[j].g() @  u2.value) #- dh_dxi @ robots[j].U  # need best case U here. not previous U
                    
                    robots[j].trust_robot[0,k] = compute_trust( A, b, robots[k].f() + robots[k].g() @ robots[k].U, robots[k].x_dot_nominal, h, min_dist, h_min )            
                    # if robots[j].trust_robot[0,k]<0:
                    #     print(f"{j}'s Trust of {k} robot: {best_controller.status}: {robots[j].trust_robot[0,k]}, h:{h}")
                    robots[j].robot_alpha[0,k] = robots[j].robot_alpha[0,k] + alpha_der_max * robots[j].trust_robot[0,k]
                    if (robots[j].robot_alpha[0,k]<0):
                        robots[j].robot_alpha[0,k] = 0.01
            
            # Plotting
            robots[j].adv_alphas = np.append( robots[j].adv_alphas, robots[j].adv_alpha, axis=0 )
            robots[j].trust_advs = np.append( robots[j].trust_advs, robots[j].trust_adv, axis=0 )
            robots[j].robot_alphas = np.append( robots[j].robot_alphas, robots[j].robot_alpha, axis=0 )
            robots[j].trust_robots = np.append( robots[j].trust_robots, robots[j].trust_robot, axis=0 )
            robots[j].robot_hs = np.append( robots[j].robot_hs, robots[j].robot_h, axis=0 )
            robots[j].adv_hs = np.append( robots[j].adv_hs, robots[j].adv_h, axis=0 )
                    
            # Solve for control input
            u1_ref.value = robots[j].U_ref
            cbf_controller.solve(solver=cp.GUROBI)
            if cbf_controller.status!='optimal':
                print(f"{j}'s input: {cbf_controller.status}")
            robots[j].nextU = u1.value       
            
            
            ## Default
            
            # Constraints in LP and QP are same      
            A1.value = robots_default[j].A1
            A2.value = robots_default[j].A1
            b1.value = robots_default[j].b1
            b2.value = robots_default[j].b1
            
            robots_default[j].adv_alphas = np.append( robots_default[j].adv_alphas, robots_default[j].adv_alpha, axis=0 )
            robots_default[j].trust_advs = np.append( robots_default[j].trust_advs, robots_default[j].trust_adv, axis=0 )
            robots_default[j].robot_alphas = np.append( robots_default[j].robot_alphas, robots_default[j].robot_alpha, axis=0 )
            robots_default[j].trust_robots = np.append( robots_default[j].trust_robots, robots_default[j].trust_robot, axis=0 )
            robots_default[j].robot_hs = np.append( robots_default[j].robot_hs, robots_default[j].robot_h, axis=0 )
            robots_default[j].adv_hs = np.append( robots_default[j].adv_hs, robots_default[j].adv_h, axis=0 )
            
            u1_ref.value = robots_default[j].U_ref
            cbf_controller.solve(solver=cp.GUROBI)
            if cbf_controller.status!='optimal':
                print(f"{j}'s input: {cbf_controller.status}")
            robots_default[j].nextU = u1.value     
            
        for j in range(num_adversaries):
            greedy[j].step(greedy[j].U_ref)    
            
            greedy_default[j].step(greedy_default[j].U_ref) 
        
        for j in range(num_robots):
            robots[j].step( robots[j].nextU )
            robots[j].render_plot()
            # print(f"{j} state: {robots[j].X[1,0]}, input:{robots[j].nextU[0,0]}, {robots[j].nextU[1,0]}")
            
            robots_default[j].step( robots_default[j].nextU )
            robots_default[j].render_plot()
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
axis1[0,0].plot(tp,robots[0].adv_alphas[1:,0],'r',label='Adversary')
axis1[0,0].plot(tp,robots[0].robot_alphas[1:,1],'g',label='Robot 2')
axis1[0,0].plot(tp,robots[0].robot_alphas[1:,2],'k',label='Robot 3')
axis1[0,0].set_title('Robot 1 alphas')
axis1[0,0].set_xlabel('time (s)')
axis1[0,0].legend()

# Robot 1
axis1[1,0].plot(tp,robots[1].adv_alphas[1:,0],'r',label='Adversary')
axis1[1,0].plot(tp,robots[1].robot_alphas[1:,0],'g',label='Robot 1')
axis1[1,0].plot(tp,robots[1].robot_alphas[1:,2],'k',label='Robot 3')
axis1[1,0].set_title('Robot 2 alphas')
axis1[1,0].set_xlabel('time (s)')
axis1[1,0].legend()

# Robot 2
axis1[0,1].plot(tp,robots[2].adv_alphas[1:,0],'r',label='Adversary')
axis1[0,1].plot(tp,robots[2].robot_alphas[1:,0],'g',label='Robot 1')
axis1[0,1].plot(tp,robots[2].robot_alphas[1:,1],'k',label='Robot 2')
axis1[0,1].set_title('Robot 3 alphas')
axis1[0,1].set_xlabel('time (s)')
axis1[0,1].legend()

#### TRUST ######
# Robot 0
figure2, axis2 = plt.subplots(2, 2)
axis2[0,0].plot(tp,robots[0].trust_advs[1:,0],'r',label='Adversary')
axis2[0,0].plot(tp,robots[0].trust_robots[1:,1],'g',label='Robot 2')
axis2[0,0].plot(tp,robots[0].trust_robots[1:,2],'k',label='Robot 3')
axis2[0,0].set_title('Robot 1 trust')
axis2[0,0].set_xlabel('time (s)')
axis2[0,0].legend()

# Robot 1
axis2[1,0].plot(tp,robots[1].trust_advs[1:,0],'r',label='Adversary')
axis2[1,0].plot(tp,robots[1].trust_robots[1:,0],'g',label='Robot 1')
axis2[1,0].plot(tp,robots[1].trust_robots[1:,2],'k',label='Robot 3')
axis2[1,0].set_title('Robot 2 trust')
axis2[1,0].set_xlabel('time (s)')
axis2[1,0].legend()

# Robot 2
axis2[0,1].plot(tp,robots[2].trust_advs[1:,0],'r',label='Adversary')
axis2[0,1].plot(tp,robots[2].trust_robots[1:,0],'g',label='Robot 1')
axis2[0,1].plot(tp,robots[2].trust_robots[1:,1],'k',label='Robot 2')
axis2[0,1].set_title('Robot 3 trust')
axis2[0,1].set_xlabel('time (s)')
axis2[0,1].legend()

figure9, axis9 = plt.subplots(1, 2)
axis9[0].plot(tp,robots[0].adv_alphas[1:,0],'r',label='Adversary')
axis9[0].plot(tp,robots[0].robot_alphas[1:,1],'g',label='Robot 2')
axis9[0].plot(tp,robots[0].robot_alphas[1:,2],'k',label='Robot 3')
axis9[0].set_title(r'Robot 1 $\alpha$')
axis9[0].set_xlabel('time (s)')
axis9[0].legend()

axis9[1].plot(tp,robots[0].adv_hs[1:,0],'r',label='Adversary - Proposed')
axis9[1].plot(tp,robots[0].robot_hs[1:,1],'g',label='Robot 2 - Proposed')
axis9[1].plot(tp,robots[0].robot_hs[1:,2],'k',label='Robot 3 - Proposed')
axis9[1].plot(tp,robots_default[0].adv_hs[1:,0],'r--',label=r'Adversary - fixed $\alpha$')
axis9[1].plot(tp,robots_default[0].robot_hs[1:,1],'g--',label=r'Robot - fixed $\alpha$')
axis9[1].plot(tp,robots_default[0].robot_hs[1:,2],'k--',label=r'Robot - fixed $\alpha$')
axis9[1].set_title('Robot 1 CBFs')
axis9[1].set_xlabel('time (s)')
axis9[1].legend()

figure6, axis6 = plt.subplots(1, 1)
axis6.plot(tp,robots[0].trust_advs[1:,0]/2,'r',label='Robot 1 trust of Adversary')
axis6.plot(tp,robots[1].trust_advs[1:,0]/2,'r--',label='Robot 2 trust of Adversary')
axis6.plot(tp,robots[0].trust_robots[1:,1]/2,'g',label='Robot 1 trust of 2 ')
axis6.plot(tp,robots[1].trust_robots[1:,0]/2,'g--',label='Robot 2 trust of 1 ')
axis6.plot(tp,robots[0].trust_robots[1:,2]/2,'k',label='Robot 1 trust of 3 ')
axis6.plot(tp,robots[2].trust_robots[1:,0]/2,'k--',label='Robot 3 trust of 1 ')
# axis6.legend(loc='upper right')
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
axis5.plot(tp,-robots[0].adv_hs[1:,0],'r')#,label='Adversary - Proposed')
axis5.plot(tp,-robots[0].robot_hs[1:,1],'g')#,label='Robot 2 - Proposed')
axis5.plot(tp,-robots[0].robot_hs[1:,2],'k')#,label='Robot 3 - Proposed')
axis5.plot(tp,-robots_default[0].adv_hs[1:,0],'r--')#,label=r'Adversary - fixed $\alpha$')
axis5.plot(tp,-robots_default[0].robot_hs[1:,1],'g--')#,label=r'Robot - fixed $\alpha$')
axis5.plot(tp,-robots_default[0].robot_hs[1:,2],'k--')#,label=r'Robot - fixed $\alpha$')
# plt.legend(loc='upper right')
# axis5.legend(loc='upper right')
axis5.set_xlabel('time (s)')
# plt.show()

figure8, axis8 = plt.subplots(1, 1)
axis8.plot(tp,robots[0].adv_alphas[1:,0],'r',label='Adversary')
axis8.plot(tp,robots[0].robot_alphas[1:,1],'g',label='Robot 2')
axis8.plot(tp,robots[0].robot_alphas[1:,2],'k',label='Robot 3')
axis8.set_title(r'Robot 1 $\alpha$')
axis8.set_xlabel('time (s)')
axis8.legend()

figure9, axis9 = plt.subplots(1, 1)
axis9.plot(tp,robots[1].adv_alphas[1:,0],'r',label='Adversary')
axis9.plot(tp,robots[1].robot_alphas[1:,0],'g',label='Robot 1')
axis9.plot(tp,robots[1].robot_alphas[1:,2],'k',label='Robot 3')
axis9.set_title(r'Robot 2 $\alpha$')
axis9.set_xlabel('time (s)')
axis9.legend()

figure10, axis10 = plt.subplots(1, 1)
axis10.plot(tp,robots[2].adv_alphas[1:,0],'r',label='Adversary')
axis10.plot(tp,robots[2].robot_alphas[1:,0],'g',label='Robot 1')
axis10.plot(tp,robots[2].robot_alphas[1:,1],'k',label='Robot 2')
axis10.set_title(r'Robot 3 $\alpha$')
axis10.set_xlabel('time (s)')
axis10.legend()

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