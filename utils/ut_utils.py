import numpy as np
import torch
from utils.sqrtm import sqrtm
from utils.identity_map import identity
from inliner import inline
from numba import njit
from numba import jit

def get_mean_cov(sigma_points, weights, compute_cov=True):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = torch.sum( weighted_points, 1 ).reshape(-1,1)
    if compute_cov==False:
        return mu
    
    # covariance
    centered_points = sigma_points - mu
    weighted_centered_points = centered_points * weights[0] 
    cov = torch.matmul( weighted_centered_points, centered_points.T )
    
    # print(f"Checking for Nans: {torch.isnan(cov).any()}")
    return mu, cov

def generate_sigma_points( mu, cov, num_sigma_points = 1 ):
    # no of points
    n = num_sigma_points #mu.shape[0]  # dimension of single vector
    N = 2*n + 1 # total points
    
    # TODO
    # k = n - 3
    k = 1
    root_term = sqrtm((n+k)*cov)
    
    if np.linalg.det( cov.detach().numpy() )< 0.01:
        root_term = cov
    else:
        root_term = sqrtm((n+k)*cov)
    
    # print(f"cov: {cov}, det: {torch.det(cov)}")
    
    new_points = torch.clone(mu)
    new_points = torch.cat( (new_points, mu - root_term[:,1].reshape(-1,1)), 1 )
    new_points = torch.cat( (new_points, mu + root_term[:,0].reshape(-1,1)), 1 )
    
    new_weights = torch.tensor([k/(n+k)]).reshape(-1,1)
    new_weights = torch.cat( (new_weights, torch.tensor(1.0/2/(n+k)).reshape(-1,1)), axis=1 )
    new_weights = torch.cat( (new_weights, torch.tensor(1.0/2/(n+k)).reshape(-1,1)), axis=1 )
            
    return new_points, new_weights
    
# def leader_predict(t):
#     uL = 0.5
#     vL = 3*np.sin(np.pi*t*4) #  0.1 # 1.2
#     # uL = 1
#     # vL = 1
#     return uL, vL
    # return leader_motion(t)

def sigma_point_expand(robot_state, sigma_points, weights, leader, cur_t = 0):
    # find number of sigma points
    n, N = sigma_points.shape
    new_points = []#torch.zeros((n,1))
    new_weights = []#np.array([0])
    for i in range(N):
        #get GP gaussian

        # sys_state = torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 )
        sys_state = sigma_points[:,i].reshape(-1,1).T
        # sys_state.retain_grad()
        mu, cov = leader.gp.predict_torch( sys_state ) # all are tensors here
        mu, cov = leader.predict_function(cur_t)  
        
        # print(f"mu:{mu}, cov:{cov}")
        
        mu = mu.reshape(-1,1)
        # mu = torch.tensor([[0.5],[0.5]]) * torch.norm( sys_state )
        # cov = torch.tensor([[0.0181, 0.0064],
        # [0.0064, 0.0282]])
        
        # TODO
        k = n - 3
        root_term =  sqrtm((n+k)*cov)
        
        if np.linalg.det( cov.detach().numpy() )< 0.01:
            root_term = cov
        else:
            root_term = sqrtm((n+k)*cov)
        
        # Now get 3 points
        if new_points==[]:
            new_points = torch.clone(mu)
        else:
            new_points = torch.cat((new_points, mu), axis=1 )
        new_points = torch.cat((new_points, mu - root_term[:,0].reshape(-1,1)), axis=1 )
        new_points = torch.cat((new_points, mu + root_term[:,1].reshape(-1,1)), axis=1 )
        if new_weights==[]:
            new_weights = weights[0,i].reshape(-1,1) * 1.0/3
        else:
            new_weights = torch.cat( (new_weights, weights[0,i].reshape(-1,1) * 1.0/3) , 1 )
        new_weights = torch.cat( (new_weights, weights[0,i].reshape(-1,1) * 1.0/3) , 1 )
        new_weights = torch.cat( (new_weights, weights[0,i].reshape(-1,1) * 1.0/3), 1 )       
        # print("new_points",new_points)
    return new_points, new_weights

def sigma_point_compress( sigma_points, weights ):
    mu, cov = get_mean_cov( sigma_points, weights )
    return generate_sigma_points( mu, cov )

def sigma_point_scale_up( sigma_points, weights, scale_factor=3 ):
    n, N = sigma_points.shape
    new_points = []#torch.zeros((n,1))
    new_weights = []#np.array([0])
    for i in range(N):
                
        if new_points==[]:
            new_points = sigma_points[:,i].reshape(-1,1).repeat( (1,scale_factor) )
        else:
            new_points = torch.cat((new_points, sigma_points[:,i].reshape(-1,1).repeat( (1,scale_factor) )), axis=1 )
        
        if new_weights==[]:
            new_weights = weights[0,i].repeat( (1,scale_factor) )/ scale_factor
        else:
            new_weights = torch.cat( (new_weights, weights[0,i].repeat( (1,scale_factor) )/scale_factor  ), axis=1 )
              
    return new_points, new_weights
    

def initialize_sigma_points(robot, num_points):
    # return 2N + 1 points
    sigma_points = torch.clone( robot.X_torch )
    for _ in range(num_points):
        sigma_points = torch.cat( (sigma_points, torch.clone( robot.X_torch )) , 1)
        sigma_points = torch.cat( (sigma_points, torch.clone( robot.X_torch )) , 1)
    weights = torch.ones((1,2*num_points +1)) * 1.0/( 2*num_points+1 )
    return sigma_points, weights

def dh_dxijk( robotJ, robotJ_state, robotK_state, robotK_type='SingleIntegrator2D', d_min = 0.3 ):
    h, dh_dxj, dh_dxk = robotJ.agent_barrier_torch(robotJ_state, robotK_state, d_min, robotK_type)
    return h, dh_dxj, dh_dxk

# @torch.jit.script
def cbf_condition_evaluator( robotJ, robotJ_state, robotK_state, robotK_state_dot, robotK_type='SingleIntegrator2D'):
    h, dh_dxj, dh_dxk = robotJ.agent_barrier_torch(robotJ_state, robotK_state, robotJ.d_min, robotK_type)    
    B = dh_dxj @ robotJ.f_torch( robotJ_state ) + dh_dxk @ robotK_state_dot + robotJ.alpha_torch @ h
    A = dh_dxj @ robotJ.g_torch( robotJ_state ) 
    return A, B

def cbf_fov_condition_evaluator( robotJ, robotJ_state, robotK_state, robotK_state_dot, robotK_type='SingleIntegrator2D'):
    h1, dh1_dxj, dh1_dxk, h2, dh2_dxj, dh2_dxk, h3, dh3_dxj, dh3_dxk = robotJ.agent_fov_barrier(robotJ_state, robotK_state, robotK_type)    
    
    B1 = dh1_dxj @ robotJ.f_torch( robotJ_state ) + dh1_dxk @ robotK_state_dot + robotJ.alpha_torch[0] * h1
    A1 = dh1_dxj @ robotJ.g_torch( robotJ_state ) 
    
    B2 = dh2_dxj @ robotJ.f_torch( robotJ_state ) + dh2_dxk @ robotK_state_dot + robotJ.alpha_torch[1] * h2
    A2 = dh2_dxj @ robotJ.g_torch( robotJ_state ) 
    
    B3 = dh3_dxj @ robotJ.f_torch( robotJ_state ) + dh3_dxk @ robotK_state_dot + robotJ.alpha_torch[2] * h3
    A3 = dh3_dxj @ robotJ.g_torch( robotJ_state ) 
    
    B = torch.cat( (B1, B2, B3), dim = 0 )
    A = torch.cat( (A1, A2, A3), dim = 0 )
    
    return A, B

# @jit
def clf_condition_evaluator( robotJ, robotJ_state, robotK_state, robotK_state_dot, robotK_type='SingleIntegrator2D' ):
    V, dV_dxj, dV_dxk = robotJ.lyapunov_tensor( robotJ_state, robotK_state )
    
    B = - dV_dxj @ robotJ.f_torch( robotJ_state ) - dV_dxk @ robotK_state_dot - robotJ.k_torch * V
    A = - dV_dxj @ robotJ.g_torch( robotJ_state )
    
    return A, B 


def clf_cbf_fov_evaluator( robotJ, robotJ_state, robotK_state, robotK_state_dot, robotK_type='SingleIntegrator2D' ):
    
    A1, B1 = clf_condition_evaluator( robotJ, robotJ_state, robotK_state, robotK_state_dot, robotK_type )
    A2, B2 = cbf_fov_condition_evaluator( robotJ, robotJ_state, robotK_state, robotK_state_dot, robotK_type )
    
    # V, dV_dxj, dV_dxk = robotJ.lyapunov_tensor( robotJ_state, robotK_state )
    
    # B1 = - dV_dxj @ robotJ.f_torch( robotJ_state ) - dV_dxk @ robotK_state_dot - robotJ.k_torch * V
    # A1 = - dV_dxj @ robotJ.g_torch( robotJ_state )
    
    # h1, dh1_dxj, dh1_dxk, h2, dh2_dxj, dh2_dxk, h3, dh3_dxj, dh3_dxk = robotJ.agent_fov_barrier(robotJ_state, robotK_state, robotK_type)    
    
    # B21 = dh1_dxj @ robotJ.f_torch( robotJ_state ) + dh1_dxk @ robotK_state_dot + robotJ.alpha_torch[0] * h1
    # A21 = dh1_dxj @ robotJ.g_torch( robotJ_state ) 
    
    # B22 = dh2_dxj @ robotJ.f_torch( robotJ_state ) + dh2_dxk @ robotK_state_dot + robotJ.alpha_torch[1] * h2
    # A22 = dh2_dxj @ robotJ.g_torch( robotJ_state ) 
    
    # B23 = dh3_dxj @ robotJ.f_torch( robotJ_state ) + dh3_dxk @ robotK_state_dot + robotJ.alpha_torch[2] * h3
    # A23 = dh3_dxj @ robotJ.g_torch( robotJ_state ) 
    
    # B2 = torch.cat( (B21, B22, B23), dim = 0 )
    # A2 = torch.cat( (A21, A22, A23), dim = 0 )
    
    
    B = torch.cat( (B1, B2), dim=0 )
    A = torch.cat( (A1, A2), dim=0 )
    
    return A, B
    

def UT_Mean_Evaluator(  fun_handle, robotJ, robotJ_state, robotK_sigma_points, robotK_dot_sigma_points, robotK_weights ):
    
    mu_A = []
    mu_B = []
    for i in range(robotK_sigma_points.shape[1]):
        A, B = fun_handle( robotJ, robotJ_state, robotK_sigma_points[:,i].reshape(-1,1), robotK_dot_sigma_points[:,i].reshape(-1,1), robotK_type='SingleIntegrator2D' )
        if mu_A==[]:
            mu_A = A * robotK_weights[0,i]
            mu_B = B * robotK_weights[0,i]
        else: 
            mu_A = mu_A + A * robotK_weights[0,i]
            mu_B = mu_B + B * robotK_weights[0,i]
    return mu_A, mu_B

# @inline 
def UT_Mean_Evaluator_basic(fun_handle, robotJ, robotK_sigma_points, robotK_weights):
    mu = []
    for i in range(robotK_sigma_points.shape[1]):
        if mu==[]:
            mu = fun_handle( robotJ, robotK_sigma_points[:,i].reshape(-1,1)  ) *  robotK_weights[0,i]
        else:
            mu = mu + fun_handle( robotJ, robotK_sigma_points[:,i].reshape(-1,1)  ) *  robotK_weights[0,i]
    return mu
