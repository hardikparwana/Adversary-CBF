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
import gpytorch

plt.rcParams.update({'font.size': 27})

# Learning algorithm for each step  ##############################################################
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# "Loss" for GPs - the marginal log likelihood
training_iter = 10

####################################################################################################

# Helper Functions to solve QP #####################################################################
def getGrad(param):
            if param.grad==None:
                print("Grad NONE")
                return 0
            value = param.grad.detach().numpy()
            param.grad = None
            return value  
            
def compute_A1_b1_tensor(robotsJ, robotsK, alpha, d_min):
    h, dh_dxj, dh_dxk = robots[j].agent_barrier_torch(robotsJ.X_torch, robotsK.X_torch, d_min, robotsK.type)
    A1 = dh_dxj @ robotsJ.g_torch(robotsJ.X_torch)
    b1 = -dh_dxj @ robotsJ.f_torch(robotsJ.X_torch) - dh_dxk @ ( robotsK.f_torch(robotsK.X_torch) + robotsK.g_torch(robotsK.X_torch) @ torch.tensor(robotsK.U,dtype=torch.float) ) - alpha * h           
    return A1, b1

def get_input(robots,j,num_robots):
    robots[j].A1 = []                
    for k in range(num_robots):
        if k==j:
            continue

        # Get constraints
        A1k, b1k = compute_A1_b1_tensor( robots[j], robots[k], robots[j].alpha_torch[k], d_min )
        if robots[j].A1 == []:
            robots[j].A1 = A1k
            robots[j].b1 = b1k
        else:
            robots[j].A1 = torch.cat( ( robots[j].A1, A1k ) , 0 )
            robots[j].b1 = torch.cat( ( robots[j].b1, b1k ) , 0 )
    
    # print(f"j: {j}, A1:{robots[j].A1}, b1:{robots[j].b1}")
    # solve QP now
    try:
        solution,  = cbf_controller_step1_layer( robots[j].U_ref_torch, robots[j].A1, robots[j].b1 )
        # print(f"solved u:{solution}")
        return solution
    except Exception as e:
        # print("QP not solvable.")
        print(e)
        exit()
        
def get_GP_model(train_x, train_y, likelihood, training_iter):
    
    model = GPModel(train_x, train_y, likelihood)
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model

def get_adversary_input(robots,j):
    target_type = type(robots[j].target)
    target = robots[j].target
    
    if target_type==int:
            V_nominal, dV_dx_nominal = robots[j].lyapunov( robots[target].X  )
            return -1.0 * dV_dx_nominal.T /np.linalg.norm( dV_dx_nominal )         
    elif target_type == str:            
            if target == 'move right':
                    return np.array([1.0, 0.0]).reshape(-1,1)
            if target == 'move left':
                    return np.array([-1.0, 0.0]).reshape(-1,1)
            if target == 'move down':
                    return np.array([0.0, -1.0]).reshape(-1,1)
            if target == 'move up':
                    return np.array([0.0,  1.0]).reshape(-1,1)
                
            
        
#####################################################################################################

# Sim Parameters                  
dt = 0.05
tf = 5.4#8#4.1 #0.2#4.1
num_steps = int(tf/dt)
outer_loop = 5
dt_outer = dt * outer_loop
t = 0
d_min = 1.0#0.1
lr_alpha = 0.2

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

alpha = 0.1

default_plot = False
save_plot = False
movie_name = 'test0_default.mp4'

# agents
robots = []
num_robots = 6
robots.append( Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf ) )
robots.append( Unicycle(np.array([2.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0, alpha=alpha_cbf ) )
robots.append( Unicycle(np.array([3.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0, alpha=alpha_cbf ) )

robots.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=1.0, target = 0) )
robots.append( SingleIntegrator2D(np.array([0,5]), dt, ax, color='r',palpha=1.0, target = 'move right') )
robots.append( SingleIntegrator2D(np.array([7,7]), dt, ax, color='r',palpha=1.0, target = 'move left') )
# agent nominal version
robots_nominal = []

robots_nominal.append( Unicycle(np.array([3,1.5,np.pi/2]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=alpha) )
robots_nominal.append( Unicycle(np.array([2.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=alpha ) )
robots_nominal.append( Unicycle(np.array([3.5,0,np.pi/2]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=1.0) )
robots_nominal.append( SingleIntegrator2D(np.array([0,5]), dt, ax, color='r',palpha=1.0) )
robots_nominal.append( SingleIntegrator2D(np.array([7,7]), dt, ax, color='r',palpha=1.0) )

U_nominal = np.zeros((2,num_robots))


# Adversarial agents
# adversary = []
# adversary.append( SingleIntegrator2D(np.array([0,4]), dt, ax) )


############################## Optimization problems ######################################

###### 1: CBF Controller: centralized
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1 
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


tp = [] 

for i in range(num_steps):
    
    ## Low frequency operation
    if i % outer_loop != 0:
        print("High frequency update")
        # higher-frequency update loop
        
        for j in range(num_robots):   

            # Initialization of tensors: get all untouched tensors now
            for k in range(num_robots):
                robots[k].X_torch = torch.tensor(robots[k].X, requires_grad=True, dtype=torch.float)
                robots[k].alpha_torch = torch.tensor(robots[j].alpha, requires_grad=True, dtype=torch.float)
                       
            # Get nominal control input and constraints
            if robots[j].identity=='adversary':
                robots[j].U = get_adversary_input(robots,j)
            elif robots[j].identity=='nominal':
                print(robots[j].identity)
                # nominal input
                robots[j].U_ref = np.array([0.0,0.0]).reshape(-1,1)
                robots[j].U_ref_torch = torch.tensor(robots[j].U_ref,dtype=torch.float)
                
                Ut = get_input(robots,j,num_robots)
                robots[j].U = Ut.detach().numpy()
            
            robots[j].Xold = robots[j].X
            
            robots[j].step(robots[j].U, dt)
            robots[j].render_plot()    
            
            # Store Data for GP fit
            xdot = (robots[j].X - robots[j].Xold)/dt
            robots[j].Xs = np.append( robots[j].Xs, robots[j].X, axis=1 )
            robots[j].Xdots = np.append( robots[j].Xdots, xdot, axis=1 )  

    else:
        print("low frequency update")
        if i<outer_loop or i==0:
            continue
        # Low frequency update
        # Fit a GP to estimate motion models
        train_x = np.copy(robots[0].Xs.T)
        for j in range(1, num_robots):
            train_x = np.append( train_x, robots[j].Xs.T, axis=1 )
        train_x = train_x[1:,:]
        train_x = torch.tensor(train_x)

        # Predict and solve QPs for two time steps for EACH robot        
        for j in range(num_robots):
            train_y = np.copy(robots[j].Xdots[:,1:].T)
            train_y = torch.tensor(train_y)   
            robots[j].gp_x = get_GP_model(train_x, train_y[:,0], likelihood, training_iter)
            robots[j].gp_y = get_GP_model(train_x, train_y[:,1], likelihood, training_iter)
            if robots[j].type=='Unicycle':
                robots[j].gp_yaw = get_GP_model(train_x, train_y[:,2], likelihood, training_iter)
            
        # Initialization of tensors: get all untouched tensors now
        for j in range(num_robots):
            
            # Get nominal control input and constraints
            if robots[j].identity=='adversary':
                # robots[j].U_ref = get_adversary_input(robots,j)
                continue 
            
            for k in range(num_robots):
                robots[k].X_torch = torch.tensor(robots[k].X, requires_grad=True, dtype=torch.float)
                robots[k].alpha_torch = torch.tensor(robots[k].alpha, requires_grad=True, dtype=torch.float)            
        
             
            targetX = torch.tensor([ [2.0],[2.0] ], dtype=torch.float)

            # nominal input
            robots[j].U_ref = np.array([0.0,0.0]).reshape(-1,1)
            robots[j].U_ref_torch = torch.tensor(robots[j].U_ref,dtype=torch.float)

            # Step 1
            Ut = get_input(robots,j,num_robots)
            print("updated")
            robots[j].X_torch = robots[j].step_torch( robots[j].X_torch,Ut, dt_outer )
            reward1 = robots[j].compute_reward(robots[j].X_torch, targetX)
            
            # nominal input
            robots[j].U_ref = np.array([0.0,0.0]).reshape(-1,1)
            robots[j].U_ref_torch = torch.tensor(robots[j].U_ref,dtype=torch.float)
            
            # Step 2
            Utplus1 = get_input(robots,j,num_robots)
            robots[j].X_torch = robots[j].step_torch( robots[j].X_torch,Utplus1, dt_outer )
            reward2 = robots[j].compute_reward(robots[j].X_torch, targetX)                

            # total reward
            reward = reward1 + reward2
            reward.sum().backward()
    
            # get parameter gradients
            alpha_grad = getGrad(robots[j].alpha_torch)
                
            # Update alpha now
            robots[j].alpha = robots[j].alpha + lr_alpha * alpha_grad.reshape(-1,1)
            
    t = t + dt
    tp.append(t)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
plt.ioff()   
