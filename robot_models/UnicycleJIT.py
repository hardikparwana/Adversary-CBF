import numpy as np
import torch
from utils.utilsJIT import *

@torch.jit.script    
def unicycle_f_torch_jit(x):
    return torch.tensor([0.0,0.0,0.0],dtype=torch.float).reshape(-1,1)

@torch.jit.script
def unicycle_g_torch_jit(x):
    # return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])
    g1 = torch.cat( (torch.cos(x[2,0]).reshape(-1,1),torch.tensor([[0]]) ), dim=1 )
    g2 = torch.cat( ( torch.tensor([[0]]), torch.sin(x[2,0]).reshape(-1,1) ), dim=1 )
    g3 = torch.tensor([[0,1]],dtype=torch.float)
    gx = torch.cat((g1,g2,g3))
    return gx

@torch.jit.script
def cat_tensors():
    return  torch.cat( [torch.tensor(1), torch.tensor(2) ], dim = 0)

@torch.jit.script
def sigma_torch(s):
    k1 = 2.0
    return torch.div( (torch.exp(k1-s)-1) , (torch.exp(k1-s)+1) )

@torch.jit.script
def sigma_der_torch(s):
    k1 = 2
    return -torch.div ( torch.exp(k1-s),( 1+torch.exp( k1-s ) ) ) @ ( 1 + sigma_torch(s) )

@torch.jit.script
def unicycle_SI2D_lyapunov_tensor_jit(X, G):
    min_D = 0.3
    max_D = 2.0
    avg_D = (min_D + max_D)/2.0
    V = torch.square ( torch.norm( X[0:2] - G[0:2] ) - avg_D )
    
    factor = 2*(torch.norm( X[0:2]- G[0:2] ) - avg_D)/torch.norm( X[0:2] - G[0:2] ) * (  X[0:2] - G[0:2] ).reshape(1,-1) 
    dV_dxi = torch.cat( (factor, torch.tensor([[0]])), dim  = 1 )
    dV_dxj = -factor
    
    return V, dV_dxi, dV_dxj

@torch.jit.script
def unicycle_SI2D_barrier_torch_jit(X, targetX): # target is unicycle
    beta = 1.01
    min_D = 0.3
    h = beta*min_D**2 -  torch.square( torch.norm(X[0:2] - targetX[0:2])  )
    h1 = h
    
    theta = X[2,0]
    s = (X[0:2] - targetX[0:2]).T @ torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    h_final = h - sigma_torch(s)
    # print(f"h1:{h1}, h2:{h}")
    # assert(h1<0)
    der_sigma = sigma_der_torch(s)
    dh_dxi =  torch.cat( ( -2*( X[0:2] - targetX[0:2] ).T - der_sigma @ torch.cat( (torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1)), dim = 1 ) ,  - der_sigma * ( torch.cos(X[2,0]).reshape(-1,1) @ ( X[0,0]-targetX[0,0] ).reshape(-1,1) - torch.sin(X[2,0]).reshape(-1,1) @ ( X[1,0] - targetX[1,0] ).reshape(-1,1) ) ), dim = 1)
    
    # Unicycle only
    dh_dxj = torch.cat( ( -2*( X[0:2] - targetX[0:2] ).T + der_sigma @ torch.cat( (torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1)),1 ) , torch.tensor([[0]]) ) , 1)
    
    return -h_final, -dh_dxi, -dh_dxj
    
@torch.jit.script
def unicycle_SI2D_barrier_torch_jit(X, targetX): # target is unicycle
    beta = 1.01
    min_D = 0.3
    h = beta*min_D**2 -  torch.square( torch.norm(X[0:2] - targetX[0:2])  )
    h1 = h
    
    theta = X[2,0]
    s = (X[0:2] - targetX[0:2]).T @ torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    h_final = h - sigma_torch(s)
    # print(f"h1:{h1}, h2:{h}")
    # assert(h1<0)
    der_sigma = sigma_der_torch(s)
    dh_dxi = torch.cat( ( -2*( X[0:2] - targetX[0:2] ).T - der_sigma @ torch.cat( (torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1)),1 ) ,  - der_sigma * ( torch.cos(X[2,0]).reshape(-1,1) @ ( X[0,0]-targetX[0,0] ).reshape(-1,1) - torch.sin(X[2,0]).reshape(-1,1) @ ( X[1,0] - targetX[1,0] ).reshape(-1,1) ) ), 1)
    # dh_dxi = torch.tensor(0)
    # Unicycle only
    dh_dxj = 2*( X[0:2] - targetX[0:2] ).T # row
    
    return -h_final, -dh_dxi, -dh_dxj

@torch.jit.script
def unicycle_SI2D_fov_barrier_jit(X, targetX):
    
    # print(f"X:{X}, targetX:{targetX}")
    
    max_D = 2.0
    min_D = 0.3
    FoV_angle = 3.14157/3
    
    # Max distance
    h1 = max_D**2 - torch.square( torch.norm( X[0:2] - targetX[0:2] ) )
    dh1_dxi = torch.cat( ( -2*( X[0:2] - targetX[0:2] ), torch.tensor([[0.0]]) ), 0).T
    dh1_dxj =  2*( X[0:2] - targetX[0:2] ).T
    
    # Min distance
    h2 = torch.square(torch.norm( X[0:2] - targetX[0:2] )) - min_D**2
    dh2_dxi = torch.cat( ( 2*( X[0:2] - targetX[0:2] ), torch.tensor([[0.0]]) ), 0).T
    dh2_dxj = - 2*( X[0:2] - targetX[0:2]).T

    # Max angle
    p = targetX[0:2] - X[0:2]

    # dir_vector = torch.tensor([[torch.cos(x[2,0])],[torch.sin(x[2,0])]]) # column vector
    dir_vector = torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    bearing_angle  = torch.matmul(dir_vector.T , p )/ torch.norm(p)
    h3 = (bearing_angle - torch.cos(FoV_angle/2))/(1.0-torch.cos(FoV_angle/2))

    norm_p = torch.norm(p)
    dh3_dx = dir_vector.T / norm_p - ( dir_vector.T @ p)  * p.T / torch.pow(norm_p,3)    
    dh3_dTheta = ( -torch.sin(X[2]) * p[0] + torch.cos(X[2]) * p[1] ).reshape(1,-1)  /torch.norm(p)
    dh3_dxi = torch.cat(  ( -dh3_dx , dh3_dTheta), 1  ) /(1.0-torch.cos(FoV_angle/2))
    dh3_dxj = dh3_dx /(1.0-torch.cos(FoV_angle/2))
    
    # print(f"dist_sq:{torch.square(torch.norm( X[0:2] - targetX[0:2] ))}, h1:{h1}, h2:{h2}, h3:{h3}")
    
    return h1, dh1_dxi, dh1_dxj, h2, dh2_dxi, dh2_dxj, h3, dh3_dxi, dh3_dxj

@torch.jit.script
def unicycle_compute_reward_jit(X,targetX):
    
    max_D = 2.0
    min_D = 0.3
    FoV_angle = 3.13/3    

    p = targetX[0:2] - X[0:2]
    dir_vector = torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    bearing_angle  = torch.matmul(dir_vector.T , p )/ torch.norm(p)
    h3 = (bearing_angle - torch.cos(FoV_angle/2))/(1.0-torch.cos(FoV_angle/2))
    
    return torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) - torch.tensor((min_D+max_D)/2) ) - 2 * h3
    
def unicycle_nominal_input_tensor_jit(X, targetX):
    k_omega = 2.0 #0.5#2.5
    k_v = 2.0 #0.5
    diff = targetX[0:2,0] - X[0:2,0]

    theta_d = torch.atan2(targetX[1,0]-X[1,0],targetX[0,0]-X[0,0])
    error_theta = wrap_angle_tensor_JIT( theta_d - X[2,0] )

    omega = k_omega*error_theta 
    v = k_v*( torch.norm(diff) ) * torch.cos( error_theta )
    v = v.reshape(-1,1)
    omega = omega.reshape(-1,1)
    U = torch.cat((v,omega))
    return U

traced_unicycle_nominal_input_tensor_jit = torch.jit.trace( unicycle_nominal_input_tensor_jit, ( torch.ones(3,1), torch.ones(2,1) ) )