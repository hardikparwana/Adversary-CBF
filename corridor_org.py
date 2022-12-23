import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2Dv2 import *
from robot_models.Unicycle import *
from robot_models.obstacles import *
from utils.utils import *
from graph_utils2_org import *

from matplotlib.animation import FFMpegWriter

plt.rcParams.update({'font.size': 15}) #27

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-5,10),ylim=(-5,7)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_aspect(1)


# Sim Parameters                  
dt = 0.05
tf = 20.0 #5.4#8#4.1 #0.2#4.1
num_steps = int(tf/dt)
t = 0
d_min_obstacles = 0.6 #0.1
d_min_agents = 0.2#0.4#0.2#0.2 #0.4
d_max = 2.0

eigen_alpha = 2.0# 0.8
alpha_cbf = 2.0 #0.7 #0.8

save_plot = False
movie_name = '2_leaders.mp4'


################# Make Obatacles ###############################
obstacles = []
index = 0
x1 = -1.0#-1.0
x2 = 1.5 #1.0
radius = 0.6
y_s = 0
y_increment = 0.3
for i in range(int( 5/y_increment )):
    obstacles.append( circle( x1,y_s,radius,ax,0 ) ) # x,y,radius, ax, id
    obstacles.append( circle( x2,y_s,radius,ax,1 ) )
    y_s = y_s + y_increment

# y1 = obstacles[-1].X[1,0] 
# y2 = y1 + 3.0
# x_s = obstacles[-1].X[0,0]
# for i in range(int( 10/y_increment )):
#     obstacles.append( circle( x_s,y1,radius,ax,0 ) ) # x,y,radius, ax, id
#     obstacles.append( circle( x_s,y2,radius,ax,1 ) )
#     x_s = x_s + y_increment
    
###################################################################

numleaders = 0
num_obstacles = len(obstacles)
num_connectivity = 1

robots = []

y_offset = -1.5
robots.append( SingleIntegrator2D(np.array([0,y_offset]), dt, ax, id = 0, color='r',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([0.5,y_offset]), dt, ax, id = 1, color='r',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )

num_leaders = len(robots)
num_robots = 12 + num_leaders
robots.append( SingleIntegrator2D(np.array([0,y_offset - 1.5]), dt, ax, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([0.0,y_offset - 0.8]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([-0.3,y_offset - 0.7]), dt, ax, id = 3, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([0.5,y_offset - 1.0]), dt, ax, id = 5, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([-1,y_offset - 3.0]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([1,y_offset - 3.0]), dt, ax, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([-0.5,y_offset - 2.0]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([0.5,y_offset - 2.0]), dt, ax, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([-0.7,y_offset - 2.7]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([0.7,y_offset - 2.7]), dt, ax, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([-0.3,y_offset - 3.0]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([0.3,y_offset - 3.0]), dt, ax, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_leaders = num_leaders, num_connectivity = num_connectivity, num_obstacles=num_obstacles ) )

# num_robots = len(robots) - num_leaders

# plt.show()

# exit()
############################## Optimization problems ######################################

###### 1: CBF Controller
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1 + num_leaders + num_obstacles + num_connectivity
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
slack_constraints1 = cp.Parameter( (num_constraints1,1), value = np.zeros((num_constraints1,1)) )
const1 = [A1 @ u1 <= b1 + slack_constraints1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )

###### 3: CBF Controller relaxed
u3 = cp.Variable((2,1))
u3_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints3  = num_robots - 1 + num_leaders + num_obstacles + num_connectivity
A3 = cp.Parameter((num_constraints3,2),value=np.zeros((num_constraints3,2)))
b3 = cp.Parameter((num_constraints3,1),value=np.zeros((num_constraints3,1)))
slack_constraints3 = cp.Parameter((num_constraints3,1), value = np.zeros((num_constraints3,1)))
const3 = [A3 @ u3 <= b3 + slack_constraints3 ]
# slack_constraints3 = cp.Variable( (num_constraints3-num_connectivity-num_obstacles,1), value = np.zeros((num_constraints3-num_connectivity-num_obstacles,1)) )
# factor_matrix = np.zeros( (num_constraints3, num_constraints3 - num_connectivity - num_obstacles) )  # 8 x 8 here
# for i in range(num_obstacles, num_constraints3-1):
#     factor_matrix[i,i-num_obstacles] = 1
# const3 = [A3 @ u3 <= b3 +  factor_matrix @ slack_constraints3 ]
objective3 = cp.Minimize( cp.sum_squares( u3 - u3_ref  ) + 1000 * cp.sum_squares( slack_constraints3 ) )
cbf_controller_relaxed = cp.Problem( objective3, const3 )


###################################################################################################

      
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

tp = []

with writer.saving(fig, movie_name, 100): 
    
    for t in range(num_steps):
        
        # for i in range(num_robots):
        #     robots[i].leader_index = 1
            # if i!=3:
            #     robots[i].leader_index = 1
            # else:
            #     robots[i].leader_index = 1
        
        # with current leader index
        L = leader_weighted_connectivity_undirected_laplacian(robots, max_dist = 6.0)
        Lambda, V = laplacian_eigen( L )
        print(f" Eigen value:{ Lambda[1] }")#, s:{ rs_robust } ")
        lambda2_dx( robots, L, Lambda[1], V[:,1].reshape(-1,1) )
        
        # Get nominal inputs
        for i in range(num_robots):
            #design
            
            if i==0:
                robots[i].U_ref = np.array([0, 0.4]).reshape(-1,1)
                # robots[i].U_ref = np.array([0.5, 0.5]).reshape(-1,1)
            elif i==1:
                robots[i].U_ref = np.array([0.4, 0.2]).reshape(-1,1)
                # robots[i].U_ref = np.array([0, 1]).reshape(-1,1)
            else:              
                robots[i].U_ref = 10*robots[i].lambda2_dx.reshape(-1,1)      
                
                 
                
        # Design constraints
        for i in range(num_robots):
            
            robots[i].A1 = np.zeros((num_constraints1,2))
            robots[i].b1 = np.zeros((num_constraints1,1))
            
            if i<num_leaders: # leaders
                continue;  
            
            const_index = 0
            
            # First constraint: Connectivity constraint: h < 0 here..  h = -eigen_value
            h_lambda, h_lambda_dxi = -(Lambda[1]-6.5), -robots[i].lambda2_dx.reshape(1,-1)
            robots[i].A1[const_index,:] = h_lambda_dxi @ robots[i].g() 
            robots[i].b1[const_index] = -robots[i].eigen_alpha * h_lambda - h_lambda_dxi @ robots[i].f()
            for j in range(num_robots):
                if j==i:
                    continue
                else:
                    dLambda_dxk = robots[j].lambda2_dx.reshape(1,-1)
                    robots[i].b1[const_index] = robots[i].b1[const_index] + dLambda_dxk @ ( robots[j].f() + robots[j].g() @ robots[j].U )
            const_index = const_index + 1
            
            # Obstacle avoidance
            for j in range(num_obstacles):
                h, dh_dxi, dh_dxj = robots[i].agent_barrier(obstacles[j], d_min_obstacles)
                robots[i].obs_h[0,j] = h
                
                # Control QP constraint
                robots[i].A1[const_index,:] = dh_dxi @ robots[i].g()
                robots[i].b1[const_index] = -dh_dxi @ robots[i].f() - robots[i].obs_alpha[0,j] * h
                
                const_index = const_index + 1
            
            # collision avoidance
            for j in range(num_robots):
                
                if j==i:
                    continue
                
                h, dh_dxi, dh_dxj = robots[i].agent_barrier(robots[j], d_min_agents)
                
                # Control QP constraint
                robots[i].A1[const_index,:] = dh_dxi @ robots[i].g()
                robots[i].b1[const_index] = -dh_dxi @ robots[i].f() - robots[i].robot_alpha[0,j] * h  #- dh_dxj @ ( robots[j].f() + robots[j].g() @ robots[j].U )
                
                const_index = const_index + 1
          
        # check if feasible. if not, then select a leader based on greedy strategy: best case constraint margin to leader      
        modified = False
        for i in range(num_robots):
            
            if i < num_leaders:
                continue
            
            A1.value = robots[i].A1
            b1.value = robots[i].b1
            
            u1_ref.value = robots[i].U_ref
            cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
            
            if cbf_controller.status!='optimal':
                if robots[i].leader_index == None:
                    
                    # choose one leader only
                    # based on distance
                    if np.linalg.norm( robots[i].X-robots[0].X ) < np.linalg.norm( robots[i].X-robots[1].X ):
                        robots[i].leader_index = 0
                    else:
                        robots[i].leader_index = 1
                    print(f"i:{i} chose leader {robots[i].leader_index}")
                    modified = True
                else:
                    modified = True
                    print("Error: already selected leader but still infeasible")
                    # exit()
                    
        if modified:
            t = t - 1
            continue
            # reform the laplacian
            L = leader_weighted_connectivity_undirected_laplacian(robots, max_dist = 6.0)
            Lambda, V = laplacian_eigen( L )
            print(f" Eigen value:{ Lambda[1] }")#, s:{ rs_robust } ")
            lambda2_dx( robots, L, Lambda[1], V[:,1].reshape(-1,1) )
        
        # remake the constraint
        # for i in range(num_robots):
        #     if i<num_leaders: # leaders
        #         continue;  
            
        #     const_index = 0
            
        #     # First constraint: Connectivity constraint: h < 0 here..  h = -eigen_value
        #     h_lambda, h_lambda_dxi = -(Lambda[1]-5.0), -robots[i].lambda2_dx.reshape(1,-1)
        #     robots[i].A1[const_index,:] = h_lambda_dxi @ robots[i].g()
        #     robots[i].b1[const_index] = -robots[i].eigen_alpha * h_lambda - h_lambda_dxi @ robots[i].f()
        #     for j in range(num_robots):
        #         if j==i:
        #             continue
        #         else:
        #             robots[i].b1[const_index] = robots[i].b1[const_index] - robots[j].lambda2_dx.reshape(1,-1) @ ( robots[j].f() + robots[j].g() @ robots[j].U )
        
        # get control input
        for i in range(num_robots):
            
            if i < num_leaders:
                robots[i].nextU = robots[i].U_ref
                continue
            
            A1.value = robots[i].A1
            b1.value = robots[i].b1
            
            u1_ref.value = robots[i].U_ref
            # u1_ref.value = 10*robots[j].lambda2_dx.reshape(-1,1)
            cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
            
            robots[i].nextU = u1.value
            
            if cbf_controller.status!='optimal':
                print("Error: should not have been infeasible here")
                        
        # implement control input
        for i in range(num_robots):
            robots[i].step( robots[i].nextU )
            
        tp.append(t*dt)
        
        print("time: ",tp[-1])
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        writer.grab_frame()
        
                
                
 # Lambda, V = laplacian_eigen( L0 )        
                # print(f" Eigen value:{ Lambda[1] }")#, s:{ rs_robust } ")                
                # lambda2_dx( robots, L, Lambda[1], V[:,1].reshape(-1,1) )
                # h_lambda, h_lambda_dxi = -(Lambda[1]-5.0), -robots[i].lambda2_dx.reshape(1,-1)
                # const_index = 0
                # A1.value[const_index,:] = h_lambda_dxi @ robots[i].g()
                # b1.value[const_index] = -robots[i].eigen_alpha * h_lambda - h_lambda_dxi @ robots[i].f()
                # for j in range(num_robots):
                #     if j==i:
                #         continue
                #     else:
                #         b1.value[const_index] = b1.value[const_index] - robots[j].lambda2_dx.reshape(1,-1) @ ( robots[j].f() + robots[j].g() @ robots[j].U )
                # cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
                # print(f"Status:{cbf_controller.status}")
                
                # print("Trying to choose leader 1")
                
                # Lambda, V = laplacian_eigen( L1 )
                
                
                # print(f" Eigen value:{ Lambda[1] }")#, s:{ rs_robust } ")                
                # lambda2_dx( robots, L, Lambda[1], V[:,1].reshape(-1,1) )
                # h_lambda, h_lambda_dxi = -(Lambda[1]-5.0), -robots[i].lambda2_dx.reshape(1,-1)
                # const_index = 0
                # A1.value[const_index,:] = h_lambda_dxi @ robots[i].g()
                # b1.value[const_index] = -robots[i].eigen_alpha * h_lambda - h_lambda_dxi @ robots[i].f()
                # for j in range(num_robots):
                #     if j==i:
                #         continue
                #     else:
                #         b1.value[const_index] = b1.value[const_index] - robots[j].lambda2_dx.reshape(1,-1) @ ( robots[j].f() + robots[j].g() @ robots[j].U )
                # cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
                # print(f"Status:{cbf_controller.status}")