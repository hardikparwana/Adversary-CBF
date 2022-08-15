import numpy as np
import torch
from utils.sqrtm import sqrtm
from utils.identity_map import identity

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
    return mu, cov

def generate_sigma_points( mu, cov ):
    # no of points
    n = mu.shape[0]  # dimension of single vector
    N = 2*n + 1 # total points
    
    k = n - 3
    root_term = sqrtm((n+k)*cov)
    
    new_points = torch.clone(mu)
    new_points = torch.cat( (new_points, mu - root_term[:,1].reshape(-1,1)), 1 )
    new_points = torch.cat( (new_points, mu + root_term[:,0].reshape(-1,1)), 1 )
    
    new_weights = torch.tensor([k/n+k]).reshape(-1,1)
    new_weights = torch.cat( (new_weights, torch.tensor(1.0/2/(n+k)).reshape(-1,1)), axis=1 )
    new_weights = torch.cat( (new_weights, torch.tensor(1.0/2/(n+k)).reshape(-1,1)), axis=1 )
            
    return new_points, new_weights
    

def sigma_point_expand(robot_state, sigma_points, weights, leader):
    # find number of sigma points
    n, N = sigma_points.shape
    new_points = []#torch.zeros((n,1))
    new_weights = []#np.array([0])
    for i in range(N):
        #get GP gaussian
        # sys_state = torch.autograd.Variable(torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 ), requires_grad=True)
        sys_state = identity( torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 ) )
        # sys_state = torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 )
        # sys_state.retain_grad()
        pred = leader.likelihood(leader.gp( sys_state )) # all are tensors here
        mu = pred.mean.reshape(-1,1)# torch.cat( (pred_x.mean, pred_y.mean), 0 )
        cov = pred.covariance_matrix # = torch.diag( torch.cat( pred_x.covariance_matrix, pred_y.covariance_matrix, 1 ) )
        
        # mu = torch.tensor([[0.5],[0.5]]) * torch.norm( sys_state )
        # cov = torch.tensor([[0.0181, 0.0064],
        # [0.0064, 0.0282]])
        
        k = n - 3
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
            new_weights = weights[0,i].repeat( (1,scale_factor) )
        else:
            new_weights = torch.cat( (new_weights, weights[0,i].repeat( (1,scale_factor) )), axis=1 )
              
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

def cbf_condition_evaluator( robotJ, robotJ_state, robotK_state, robotK_state_dot, robotK_type='SingleIntegrator2D'):
    h, dh_dxj, dh_dxk = robotJ.agent_barrier_torch(robotJ_state, robotK_state, robotJ.d_min, robotK_type)    
    B = dh_dxj @ robotJ.f_torch( robotJ_state ) + dh_dxk @ robotK_state_dot + robotJ.alpha_torch @ h
    A = dh_dxj @ robotJ.g_torch( robotJ_state ) 
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

def UT_Mean_Evaluator_basic(fun_handle, robotJ, robotK_sigma_points, robotK_weights):
    mu = []
    for i in range(robotK_sigma_points.shape[1]):
        if mu==[]:
            mu = fun_handle( robotJ, robotK_sigma_points[:,i].reshape(-1,1)  ) *  robotK_weights[0,i]
        else:
            mu = mu + fun_handle( robotJ, robotK_sigma_points[:,i].reshape(-1,1)  ) *  robotK_weights[0,i]
    return mu
