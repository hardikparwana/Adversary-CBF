import numpy as np
import torch
from sqrtm import *

def get_mean_cov(sigma_points, weights, compute_cov=True):
    weighted_points = torch.mv( sigma_points, weights )
    
    # mean
    mu = torch.sum( weighted_points, 1 ).reshape(-1,1)
    if compute_cov==False:
        return mu
    # covariance
    centered_points = sigma_points - mu
    weighted_centered_points = torch.mv( centered_points, weights )
    cov = weights * torch.sum( torch.matmul( weighted_centered_points, centered_points.T ) )
    return mu, cov

def generate_sigma_points( mu, cov ):
    # no of points
    n = mu.shape[0]  # dimension of single vector
    N = 2*n + 1 # total points
    
    k = n - 3
    root_term = MatrixSquareRoot((n+k)*cov)
    
    new_points = torch.clone(mu)
    new_points = torch.cat( new_points, mu + root_term, 1 )
    new_points = torch.cat( new_points, mu - root_term, 1 )
            
    return new_points
    

def sigma_point_expand(robot_state, sigma_points, weights, leader):
    # find number of sigma points
    n, N = sigma_points.shape
    new_points = []#torch.zeros((n,1))
    new_weights = []#np.array([0])
    for i in range(N):
        #get GP gaussian
        sys_state = torch.cat( (robot_state.T, sigma_points[:,i].reshape(-1,1).T), 1 )
        pred_x = leader.gp_x( sys_state ) # all are tensors here
        pred_y = leader.gp_x( sys_state )
        mu = torch.cat( (pred_x.mean, pred_y.mean), 0 )
        sigma = torch.diag( torch.cat( pred_x.covariance_matrix, pred_y.covariance_matrix, 1 ) )
        
        k = n - 3
        root_term = MatrixSquareRoot((n+k)*cov)
        
        # Now get 3 points
        if new_points==[]:
            new_points = torch.clone(mu)
        else:
            new_points = torch.cat((new_points, mu), axis=1 )
        new_points = torch.cat((new_points, mu - root_term), axis=1 )
        new_points = torch.cat((new_points, mu + root_term), axis=1 )
        
        if new_weights==[]:
            new_weights = weights[i] * 1.0/3
        else:
            new_weights = torch.cat( (new_weights, weights[i] * 1.0/3) )
        new_weights = torch.cat( (new_weights, weights[i] * 1.0/3) )
        new_weights = torch.cat( (new_weights, weights[i] * 1.0/3) )       
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
            new_points = sigma_points[i].repeat( (1,scale_factor) )
        else:
            new_points = torch.cat((new_points, sigma_points[i].repeat( (1,scale_factor) )), axis=1 )
        
        if new_weights==[]:
            new_weights = weights[i].repeat( (1,scale_factor) )
        else:
            new_weights = torch.cat( (new_weights, weights[i].repeat( (1,scale_factor) )) )
              
    return new_points, new_weights
    

def initialize_sigma_points(robot, num_points):
    # return 2N + 1 points
    sigma_points = torch.clone( robot.X_torch )
    for _ in range(num_points):
        sigma_points = torch.cat( (sigma_points, torch.clone( robot.X_torch )) , 1)
        sigma_points = torch.cat( (sigma_points, torch.clone( robot.X_torch )) , 1)
    weights = torch.ones((1,2*num_points +1)) * 1.0/( 2*num_points+1 )
    return sigma_points, weights

def dh_dxijk( robotJ, robotJ_state, robotK_state, robotK_type="Double Integrator", d_min = 0.3 ):
    h, dh_dxj, dh_dxk = robotJ.agent_barrier_torch(robotJ_state, robotK_state, d_min, robotK_type)
    return h, dh_dxj, dh_dxk

def cbf_condition_evaluator( robotJ, robotJ_state, robotK_state, robotK_state_dot, robotK_type="DoubleIntegrator"):
    h, dh_dxj, dh_dxk = robotJ.agent_barrier_torch(robotJ_state, robotK_state, d_min, robotK_type)    
    A = dh_dxj @ robotJ.f_torch( robotJ_state ) + dh_dxk @ robotK_state_dot + robotJ.alpha_tch @ h
    B = dh_dxj @ robotJ.g_torch( robotJ_state ) 
    return A, B

def UT_Mean_Evaluator(  fun_handle, robotJ, robotJ_state, robotK_sigma_points, robotK_dot_sigma_points, robotK_weights ):
    
    mu = torch.zeros( (robotK_sigma_points.shape[0],1), dtype=torch.float )
    for i in range(robotK_sigma_points.shape[1]):
        A, B = fun_handle( robotK_sigma_points[:,i].reshape(-1,1) )
        mu_A = mu_A + A * robotK_weights[i]
        mu_B = mu_B + B * robotK_weights[i]
    return mu_A, mu_B