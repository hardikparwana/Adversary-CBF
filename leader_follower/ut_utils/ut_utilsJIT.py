import numpy as np
import torch
from utils.sqrtm import sqrtm
# from utils.symsqrt import symsqrt
from utils.identity_map import identity
from robot_models.UnicycleJIT import *
from inliner import inline

from utils.mvgp_jit import traced_predict_torch_jit

# @torch.jit.script
def get_mean_cov_JIT(sigma_points, weights):
    
    # mean
    weighted_points = sigma_points * weights[0]
    mu = torch.sum( weighted_points, 1 ).reshape(-1,1)
    
    # covariance
    centered_points = sigma_points - mu
    weighted_centered_points = centered_points * weights[0] 
    cov = torch.matmul( weighted_centered_points, centered_points.T )
    
    # print(f"Checking for Nans: {torch.isnan(cov).any()}")
    return mu, cov

# @torch.jit.script
def get_mean_JIT(sigma_points, weights):
    weighted_points = sigma_points * weights[0]
    mu = torch.sum( weighted_points, 1 ).reshape(-1,1)
    return mu

# @torch.jit.script
def generate_sigma_points_JIT( mu, cov_root ):
    
    n = mu.shape[0]     
    N = 2*n + 1 # total points

    # TODO
    # k = n - 3
    k = 2

    new_points = torch.clone(mu)
    new_weights = torch.tensor([1.0*k/(n+k)]).reshape(-1,1)#torch.tensor([k/(n+k)]).reshape(-1,1)
    for i in range(n):
        new_points = torch.cat( (new_points, mu - cov_root[:,i].reshape(-1,1)), dim = 1 )
        new_points = torch.cat( (new_points, mu + cov_root[:,i].reshape(-1,1)), dim = 1 )

        new_weights = torch.cat( (new_weights, torch.tensor(1.0/2/(n+k)).reshape(-1,1)), dim = 1 )
        new_weights = torch.cat( (new_weights, torch.tensor(1.0/2/(n+k)).reshape(-1,1)), dim = 1 )

    return new_points, new_weights

mu_t = torch.ones((2,1)).reshape(-1,1)
cov_t = torch.tensor([ [ 1.0, 0.0 ], [0.0, 3.0] ])
traced_generate_sigma_points_JIT = torch.jit.trace( generate_sigma_points_JIT, ( mu_t, cov_t ) )

def predict_function_jit(cur_t, noise): 
    # mu = torch.zeros((2,1)).reshape(-1,1)
    # cov = torch.zeros((2,2))
    # return mu, cov
    print("cur_t", cur_t)
    uL = 0.5
    vL = 3*np.sin(np.pi*cur_t*4) #  0.1 # 1.2
    mu = torch.tensor([[uL, vL]], dtype=torch.float).reshape(-1,1)
    cov = torch.zeros((2,2), dtype=torch.float)
    cov[0,0] = noise
    cov[1,1] = noise
    return mu, cov
traced_predict_function_jit = torch.jit.trace( predict_function_jit, ( torch.tensor(1), torch.tensor(0.2) ) )


def get_ut_cov_root(cov):
    k = 2
    n = cov.shape[0]
    if torch.linalg.det( cov )< 0.01:
        root_term = cov
    else:
        root_term = sqrtm((n+k)*cov)
    return root_term

def get_ut_cov_root_diagonal(cov):
    k = 2
    n = cov.shape[0]
    
    cov_abs = torch.abs(cov)
    if cov_abs[0,0]>0.01:
        root0 = torch.sqrt(cov[0,0])
    else:
        root0 = torch.tensor(0, dtype=torch.float)
    if cov_abs[1,1]>0.01:
        root1 = torch.sqrt(cov[1,1])
    else:
        root1 = torch.tensor(0, dtype=torch.float)
    
    root_term = torch.diag( (n+k) * torch.cat( ( root0.reshape(-1,1), root1.reshape(-1,1) ), dim = 1 )[0] )

    return root_term

# @torch.jit.script
def sigma_point_expand_JIT(robot_state, sigma_points, weights, cur_t, noise):
   
    n, N = sigma_points.shape
   
    # sys_state = torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 )
    sys_state = sigma_points[:,0].reshape(-1,1).T
    
    # mu, cov = leader.gp.predict_torch( sys_state ) # all are tensors here
    mu, cov = traced_predict_function_jit(cur_t, noise)  
    
    root_term = get_ut_cov_root_diagonal(cov) 
    
    # if first_generate_sigma_run:
    #     traced_generate_sigma_points_JIT = torch.jit.trace( generate_sigma_points_JIT, ( mu, root_term ) )
    temp_points, temp_weights = traced_generate_sigma_points_JIT( mu, root_term )
    new_points = torch.clone( temp_points )
    new_weights = (torch.clone( temp_weights ) * weights[0,0]).reshape(1,-1)
        
    for i in range(1,N):
        # sys_state = torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 )
        sys_state = sigma_points[:,i].reshape(-1,1).T
        
        # mu, cov = leader.gp.predict_torch( sys_state ) # all are tensors here
        mu, cov = traced_predict_function_jit(cur_t, noise)  
        
        # TODO
        root_term = get_ut_cov_root_diagonal(cov)        
        
        temp_points, temp_weights = traced_generate_sigma_points_JIT( mu, root_term )

        new_points = torch.cat((new_points, temp_points), dim=1 )
        new_weights = torch.cat( (new_weights, (temp_weights * weights[0,i]).reshape(1,-1) ) , dim=1 )
            
        # print("new_points",new_points)
    return new_points, new_weights

# @torch.jit.script
def sigma_point_compress_JIT( sigma_points, weights ):
    mu, cov = get_mean_cov_JIT( sigma_points, weights )
    # TODO send root term
    cov_root_term = get_ut_cov_root( cov )  
    return generate_sigma_points_JIT( mu, cov_root_term )

# @torch.jit.script
def sigma_point_scale_up5_JIT( sigma_points, weights ):
    scale_factor=5
    n, N = sigma_points.shape
    
    new_points = sigma_points[:,0].reshape(-1,1).repeat( (1,scale_factor) )
    new_weights = weights[0,0].repeat( (1,scale_factor) )/ scale_factor
    for i in range(1,N):                
        new_points = torch.cat((new_points, sigma_points[:,i].reshape(-1,1).repeat( [1,scale_factor] )), dim=1 )    
        new_weights = torch.cat( (new_weights, weights[0,i].repeat( [1,scale_factor] )/scale_factor  ), dim=1 )
              
    return new_points, new_weights
    
# @torch.jit.script
def initialize_sigma_points2_JIT(X):
    # return 2N + 1 points
    n = X.shape[0]
    num_points = 2*n + 1
    sigma_points = torch.clone( X )
    for _ in range(n):
        sigma_points = torch.cat( (sigma_points, torch.clone( X )) , dim = 1)
        sigma_points = torch.cat( (sigma_points, torch.clone( X )) , dim = 1)
    weights = torch.ones((1,num_points)) * 1.0/( num_points )
    return sigma_points, weights

# @torch.jit.script
def dh_dx_unicycle_SI2D_JIT( robotJ_state, robotK_state ):
    h, dh_dxj, dh_dxk = unicycle_SI2D_barrier_torch_jit(robotJ_state, robotK_state)
    return h, dh_dxj, dh_dxk

# TODO: what is alpha torch here???
# @torch.jit.script
def cbf_condition_evaluator_unicycle_SI2D( robotJ_state, robotK_state, robotK_state_dot, alpha_torch):
    h, dh_dxj, dh_dxk = unicycle_SI2D_barrier_torch_jit(robotJ_state, robotK_state)    
    B = dh_dxj @ unicycle_f_torch_jit( robotJ_state ) + dh_dxk @ robotK_state_dot + alpha_torch @ h
    A = dh_dxj @ unicycle_g_torch_jit( robotJ_state ) 
    return A, B

# @torch.jit.script
def unicycle_SI2D_cbf_fov_condition_evaluator( robotJ_state, robotK_state, robotK_state_dot, alpha_torch):
    h1, dh1_dxj, dh1_dxk, h2, dh2_dxj, dh2_dxk, h3, dh3_dxj, dh3_dxk = unicycle_SI2D_fov_barrier_jit(robotJ_state, robotK_state)    
    
    B1 = dh1_dxj @ unicycle_f_torch_jit( robotJ_state ) + dh1_dxk @ robotK_state_dot + alpha_torch[0] * h1
    A1 = dh1_dxj @ unicycle_g_torch_jit( robotJ_state ) 
    
    B2 = dh2_dxj @ unicycle_f_torch_jit( robotJ_state ) + dh2_dxk @ robotK_state_dot + alpha_torch[1] * h2
    A2 = dh2_dxj @ unicycle_g_torch_jit( robotJ_state ) 
    
    B3 = dh3_dxj @ unicycle_f_torch_jit( robotJ_state ) + dh3_dxk @ robotK_state_dot + alpha_torch[2] * h3
    A3 = dh3_dxj @ unicycle_g_torch_jit( robotJ_state ) 
    
    B = torch.cat( (B1, B2, B3), dim = 0 )
    A = torch.cat( (A1, A2, A3), dim = 0 )
    
    # print(f"h1:{h1}, , h2:{h2}, h3:{h3}")
    
    return A, B

# @torch.jit.script
def unicycle_SI2D_clf_condition_evaluator( robotJ_state, robotK_state, robotK_state_dot, k_torch ):
    V, dV_dxj, dV_dxk = unicycle_SI2D_lyapunov_tensor_jit( robotJ_state, robotK_state )
    
    B = - dV_dxj @ unicycle_f_torch_jit( robotJ_state ) - dV_dxk @ robotK_state_dot - k_torch * V
    A = - dV_dxj @ unicycle_g_torch_jit( robotJ_state )
    
    return A, B 

# @torch.jit.script
def unicycle_SI2D_clf_cbf_fov_evaluator( robotJ_state, robotK_state, robotK_state_dot, k_torch, alpha_torch ):
    
    A1, B1 = unicycle_SI2D_clf_condition_evaluator( robotJ_state, robotK_state, robotK_state_dot, k_torch )
    A2, B2 = unicycle_SI2D_cbf_fov_condition_evaluator( robotJ_state, robotK_state, robotK_state_dot, alpha_torch )   
    
    B = torch.cat( (B1, B2), dim=0 )
    A = torch.cat( (A1, A2), dim=0 )
    
    return A, B
    
# @torch.jit.script
def unicycle_SI2D_UT_Mean_Evaluator(  robotJ_state, robotK_sigma_points, robotK_dot_sigma_points, robotK_weights, k_torch, alpha_torch ):
    
    # A, B = unicycle_SI2D_clf_cbf_fov_evaluator( robotJ_state, robotK_sigma_points[:,0].reshape(-1,1), robotK_dot_sigma_points[:,0].reshape(-1,1), k_torch, alpha_torch)
    
    A1, B1 = unicycle_SI2D_clf_condition_evaluator( robotJ_state, robotK_sigma_points[:,0].reshape(-1,1), robotK_dot_sigma_points[:,0].reshape(-1,1), k_torch )
    A2, B2 = unicycle_SI2D_cbf_fov_condition_evaluator( robotJ_state, robotK_sigma_points[:,0].reshape(-1,1), robotK_dot_sigma_points[:,0].reshape(-1,1), alpha_torch )   
    
    B = torch.cat( (B1, B2), dim=0 )
    A = torch.cat( (A1, A2), dim=0 )
    
    mu_A = A * robotK_weights[0,0]
    mu_B = B * robotK_weights[0,0]
    
    for i in range(1,robotK_sigma_points.shape[1]):
        # A, B = unicycle_SI2D_clf_cbf_fov_evaluator( robotJ_state, robotK_sigma_points[:,i].reshape(-1,1), robotK_dot_sigma_points[:,i].reshape(-1,1), k_torch, alpha_torch )
        
        A1, B1 = unicycle_SI2D_clf_condition_evaluator( robotJ_state, robotK_sigma_points[:,0].reshape(-1,1), robotK_dot_sigma_points[:,0].reshape(-1,1), k_torch )
        A2, B2 = unicycle_SI2D_cbf_fov_condition_evaluator( robotJ_state, robotK_sigma_points[:,0].reshape(-1,1), robotK_dot_sigma_points[:,0].reshape(-1,1), alpha_torch )   
        
        B = torch.cat( (B1, B2), dim=0 )
        A = torch.cat( (A1, A2), dim=0 )
        
        mu_A = mu_A + A * robotK_weights[0,i]
        mu_B = mu_B + B * robotK_weights[0,i]
    return mu_A, mu_B

# @torch.jit.script
def unicycle_reward_UT_Mean_Evaluator_basic(robotJ_state, robotK_sigma_points, robotK_weights):
    mu = unicycle_compute_reward_jit( robotJ_state, robotK_sigma_points[:,0].reshape(-1,1)  ) *  robotK_weights[0,0]
    for i in range(1, robotK_sigma_points.shape[1]):
        mu = mu + unicycle_compute_reward_jit( robotJ_state, robotK_sigma_points[:,i].reshape(-1,1)  ) *  robotK_weights[0,i]
    return mu


# # @torch.jit.script
# def sigma_point_expand_JIT(robot_state, sigma_points, weights):
#     # find number of sigma points
#     n, N = sigma_points.shape
#     new_points = []#torch.zeros((n,1))
#     new_weights = []#np.array([0])
    
#     # i = 0
#     # sys_state = torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 )
#     sys_state = sigma_points[:,0].reshape(-1,1).T
#     mu = torch.tensor([[0.5],[0.5]]) * torch.norm( sys_state )
#     cov = torch.tensor([[0.0181, 0.0064],
#     [0.0064, 0.0282]])
#     root_term = cov
#     mu = mu.reshape(-1,1)
    
#     new_points = torch.clone(mu)
#     new_weights = weights[0,0].reshape(-1,1) * 1.0/3
#     for i in range(n):
#         new_points = torch.cat((new_points, mu - root_term[:,i].reshape(-1,1)), dim=1 )
#         new_points = torch.cat((new_points, mu + root_term[:,i].reshape(-1,1)), dim=1 )
#         new_weights = torch.cat( (new_weights, weights[0,i].reshape(-1,1) * 1.0/3) , dim=1 )
#         new_weights = torch.cat( (new_weights, weights[0,i].reshape(-1,1) * 1.0/3), dim=1 ) # TODO: weights wrong here 
        
#     points, weights = generate_sigma_points_JIT( mu, root_term )
#     new_points = torch.clone( points )
#     new_weights = torch.clone( weights )
        
#     for i in range(1,N):
#         #get GP gaussian

#         # sys_state = torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 )
#         sys_state = sigma_points[:,i].reshape(-1,1).T
#         # sys_state.retain_grad()
#         # mu, cov = leader.gp.predict_torch( sys_state ) # all are tensors here
#         # mu, cov = leader.predict_function(cur_t)  
#         # print(f"mu:{mu}, cov:{cov}")
        
#         # TODO
#         root_term = cov
#         # #################    TODO***********************
#         # root_term =  symsqrt((n+k)*cov)
        
#         # if np.linalg.det( cov.detach().numpy() )< 0.01:
#         #     root_term = cov
#         # else:
#         #     root_term = symsqrt((n+k)*cov)
        
#         # Now get 3 points

#         new_points = torch.cat((new_points, mu), dim=1 )
#         new_weights = torch.cat( (new_weights, weights[0,i].reshape(-1,1) * 1.0/3) , 1 )
        
#         for j in range(n):
#             new_points = torch.cat((new_points, mu - root_term[:,0].reshape(-1,1)), dim=1 )
#             new_points = torch.cat((new_points, mu + root_term[:,1].reshape(-1,1)), dim=1 )
                
#             new_weights = torch.cat( (new_weights, weights[0,i].reshape(-1,1) * 1.0/3) , dim=1 )
#             new_weights = torch.cat( (new_weights, weights[0,i].reshape(-1,1) * 1.0/3), dim=1 )       
#         # print("new_points",new_points)
#     return new_points, new_weights