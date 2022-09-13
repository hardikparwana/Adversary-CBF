import time
import torch
import numpy as np
import random
import matplotlib 
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from numba import jit
# TODO add something to diff here
# TODO: square vs constant

def clip_angle_torch(x):
    if x[2] > torch.tensor(np.pi, dtype=torch.float):
        x[2] = x[2] - torch.tensor(2 * np.pi, dtype=torch.float)
    elif x[2] < torch.tensor(-np.pi, dtype=torch.float):
        x[2] = x[2] + torch.tensor(2 * np.pi, dtype=torch.float)
    return x

def evaluate_kernel_torch_gaussian_jit(sigma_torch, l_torch, L_torch, p_torch, x1, x2):
    diff = torch.norm((x1-x2)/l_torch)
    return sigma_torch * torch.exp(-diff**2 / 2.0 ) # / torch.square(l_torch))
    
def evaluate_kernel_torch_periodic_jit(sigma_torch, l_torch, L_torch, p_torch, x1, x2):
    diff = torch.norm(x1 - x2)
    return torch.exp( -2/L_torch**2 * ( torch.sin( np.pi * diff**2 / p_torch ))**2 )

def evaluate_kernel_torch_gaussian_periodic_jit(sigma_torch, l_torch, L_torch, p_torch, x1, x2):
    diff = torch.norm(x1 - x2)
    return sigma_torch * torch.exp(-diff**2 / (2*l_torch**2)) + torch.exp( -2/L_torch**2 * ( torch.sin( np.pi * diff**2 / p_torch ))**2 )

def get_X_cov_torch_jit(sigma_torch, l_torch, L_torch, p_torch, X_obs_torch, Xnew):
    N = X_obs_torch.shape[0]
    # print("***************************88X_obs_torch shape", X_obs_torch.shape)
    K_star_torch = torch.zeros( (N,1), dtype=torch.float )
    for i in range(N):
        K_star_torch[i,0] = evaluate_kernel_torch_gaussian_jit(sigma_torch, l_torch, L_torch, p_torch, X_obs_torch[i,:], Xnew)
    return K_star_torch

def predict_torch_jit(GA, PE, sigma_torch, l_torch, L_torch, p_torch, omega_torch, Xnew, K_inv_torch, X_obs_torch, Y_obs_torch, noise, Kcov ):
    
    k_star = traced_get_X_cov_torch_jit(sigma_torch, l_torch, L_torch, p_torch, X_obs_torch, Xnew)
    mean =  (K_inv_torch @ k_star).T @ Y_obs_torch
    
    Sigma = evaluate_kernel_torch_gaussian_jit(sigma_torch, l_torch, L_torch, p_torch, Xnew, Xnew) + noise - (K_inv_torch @ k_star).T @ torch.clone(k_star)
    cov = torch.kron(Sigma, omega_torch)
    
    # mean = torch.tensor([[1,1]], dtype=torch.float)
    # cov = torch.zeros((2,2), dtype=torch.float)
    # if cov<0:
    #     print("ERROR in cov **************************************************", cov)
    #     cov = cov + torch.tensor([[0.1]])
    #     cov = torch.tensor([[0]])
    return mean, cov

traced_size = 30
traced_predict_torch_jit = predict_torch_jit #torch.jit.trace( predict_torch_jit, ( torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor([[1.0]]), torch.ones((1,5)), torch.eye(traced_size),  torch.ones((traced_size,5)) , torch.ones((traced_size,1)), torch.tensor(0.0) ) )
traced_get_X_cov_torch_jit = get_X_cov_torch_jit #torch.jit.trace( get_X_cov_torch_jit, (torch.tensor(2.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), torch.ones((traced_size,5)), torch.ones((1,5))) )


# select a subset for predictions
# @jit(nopython=True)
def resample_obs_numba( X, Y, n_samples=80, start_index = 0 ):
    N = X.shape[0]
    idx = random.sample(range(0, N), min(n_samples, N))
    X_obs, Y_obs = X[idx,:], Y[idx,:]
    N_data = X_obs.shape[0]
    return X_obs, Y_obs, N

# @jit(nopython=True)
def evaluate_kernel_numba(GA, PE, sigma, l, L, p, x1, x2):
    return evaluate_kernel_gaussian_numba( GA, PE, sigma, l, L, p, x1, x2 )
       
def clip_angle( x ):
    # if x[2] > np.pi:
    #     x[2] = x[2] - 2 * np.pi
    # elif x[2] < -np.pi:
    #     x[2] = x[2] + 2* np.pi
    return x       

# @jit(nopython=True) 
def evaluate_kernel_gaussian_numba(GA, PE, sigma, l, L, p, x1, x2):
    diff = np.linalg.norm(clip_angle(x1-x2)/l)
    # print(f"diff:{diff}, return:{ sigma**2 * np.exp(-diff**2 / (2*l**2)) }")
    return GA * sigma * np.exp(-diff**2 / 2.0) #(2 * l**2))

# @jit(nopython=True)    
def evaluate_kernel_periodic_numba(GA, PE, sigma, l, L, h, x1, x2):
    diff = np.linalg.norm(x1 - x2)
    return PE * np.exp( -2/L**2 * ( np.sin( np.pi * diff**2 / p ))**2 )

# @jit(nopython=True)
def evaluate_kernel_gaussian_periodic_numba(GA, PE, sigma, l, L, p, x1, x2):
    diff = np.linalg.norm(x1 - x2)
    return GA * sigma**2 * np.exp(-diff**2 / (2*l**2)) + PE * np.exp( -2/L**2 * ( np.sin( np.pi * diff**2 / p ))**2 )

# @jit(nopython=True)    
def get_X_cov_numba(GA, PE, sigma, l, L, p, X_obs, Xnew):
    N_data = X_obs.shape[0]
    K_star = np.zeros( (N_data,1))
    for i in range(N_data):
        K_star[i,0] = evaluate_kernel_gaussian_numba(GA, PE, sigma, l, L, p, X_obs[i,:], Xnew)
    return K_star

# @jit(nopython=True)
def get_obs_covariance_numba(N, x, X_obs, noise = 0.1):
        x = X_obs[-1,:]
        K_obs = np.zeros((N,N))
        for i in range(N):
            val = evaluate_kernel_numba(x, X_obs[i,:])
            if (i == N-1):
                K_obs[N-1, N-1] = val + noise
            else:
                K_obs[i, N-1] = val
                K_obs[N-1, i] = val
        return K_obs[0:N, 0:N]
    
# @jit(nopython=True)
def predict_numba(GA, PE, sigma, l, L, p, omega, noise, X_obs, Y_obs, K_inv, Xnew):
    
    k_star = get_X_cov_numba(GA, PE, sigma, l, L, p, X_obs, Xnew)
    mean =  (K_inv @ k_star).T @ Y_obs
    
    Sigma = evaluate_kernel_gaussian_numba(GA, PE, sigma, l, L, p, Xnew, Xnew) - (K_inv @ k_star).T @ k_star + noise
    cov = np.kron(Sigma, omega)
    
    # mean = torch.tensor([[1,1]], dtype=torch.float)
    # cov = torch.zeros((2,2), dtype=torch.float)
    return mean, cov

# @jit(nopython=True)    
def resample_numba(X, Y, n_samples=80, start_index = 0):
        N = X.shape[0]
        idx = random.sample(range(0, N), min(n_samples, N))
        X_s, Y_s = X[idx], Y[idx]
        return X_s, Y_s
    
# Evaluate derivative of kernel (w.r.t. length scale)
# @jit(nopython=True)
def dk_dl_numba(GA, PE, sigma, l, L, p, omega, x1, x2):
    diff = np.linalg.norm(x1 - x2)
    return GA * sigma**2 * np.exp(-diff**2 / (2*l**2)) * (diff**2 / (l**3))

# Evaluate derivative of  kernel (w.r.t. sigma)
# @jit(nopython=True)
def dk_ds_numba(GA, PE, sigma, l, L, p, omega, x1, x2):
    diff = np.linalg.norm(x1 - x2)
    return GA * 2*sigma * np.exp(-diff**2 / (2*l**2))

# @jit(nopython=True)
def dk_dL_numba(GA, PE, sigma, l, L, p, omega, x1, x2):
    diff = np.linalg.norm(x1 - x2)
    return PE * -2/L**2 * ( np.sin( np.pi * diff**2 / p ))**2 * ( -2/L**3 )

# @jit(nopython=True)    
def dk_dp_numba(GA, PE, sigma, l, L, p, omega, x1, x2):
    diff = np.linalg.norm(x1 - x2)
    return PE * -2/L**2 * ( np.sin( np.pi * diff**2 / p ))**2 * ( 2 * np.sin( np.pi * diff**2 / p ) * np.cos( np.pi * diff**2 / p ) * 2 * np.pi * diff**2 * (-1/p**2) )

# @jit(nopython=True)    
def get_covariance_numba(GA, PE, sigma, l, L, p, omega,  noise, X_s):
        N = len(X_s)
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                val = evaluate_kernel_numba(GA, PE, sigma, l, L, p, X_s[i,:], X_s[j,:])
                if (i == j):
                    K[i, i] = val.item() #+ noise
                else:
                    K[i, j] = val.item()
                    K[j, i] = val.item()
        
        return K
    
def get_covariance_inv_numba( GA, PE, sigma, l, L, p, omega,  noise, X_s ):
    cov = get_covariance_numba(GA, PE, sigma, l, L, p, omega,  noise, X_s)
    cov_inv = np.linalg.inv( cov )
    return cov, cov_inv
    
# Get derivative of covariance matrix (w.r.t. length scale and sigma)
# @jit(nopython=True)
def get_dK_numba(GA, PE, sigma, l, L, p, omega, X_s):
    N = len(X_s)
    Kl = np.zeros((N, N))
    Ks = np.zeros((N, N))
    KL = np.zeros((N,N))
    Kp = np.zeros((N,N))
    
    for i in range(N):
        for j in range(i, N):
            val_l = dk_dl_numba(GA, PE, sigma, l, L, p, omega, X_s[i,:], X_s[j,:])
            val_s = dk_ds_numba(GA, PE, sigma, l, L, p, omega, X_s[i,:], X_s[j,:])
            val_L = dk_dL_numba(GA, PE, sigma, l, L, p, omega, X_s[i,:], X_s[j,:])
            val_p = dk_dp_numba(GA, PE, sigma, l, L, p, omega, X_s[i,:], X_s[j,:])
            if (i == j):
                Kl[i, i] = val_l
                Ks[i, i] = val_s
                KL[i, i] = val_L
                Kp[i, i] = val_p
            else:
                Kl[i, j] = val_l
                Kl[j, i] = val_l
                Ks[i, j] = val_s
                Ks[j, i] = val_s
                KL[i, j] = val_L
                Kp[j, i] = val_L
                KL[i, j] = val_p
                Kp[j, i] = val_p
                    
    return Kl, Ks, KL, Kp
    
# Get gradient of negative log likelihood (w.r.t. length scale, sigma, omega)
# @jit(nopython=True)
def likelihood_gradients_numba(GA, PE, sigma, l, L, p, omega, noise, X_s, Y_s, print_status = False):
    n = X_s.shape[0]
    d = Y_s.shape[1]
    
    K = get_covariance_numba(GA, PE, sigma, l, L, p, omega,  noise, X_s)               
    Kinv = np.linalg.inv(K)
    omegainv = np.linalg.inv(omega)        
    A = Kinv @ Y_s @ omegainv @ Y_s.T #  np.matmul(Kinv, np.matmul(self.Y_s, np.matmul(omegainv, np.transpose(self.Y_s))))
    
    detK = np.linalg.det(K)
    if detK < 0:
        detK = 0.0001
    Likelihood = (n*d/2) * np.log(2*np.pi) + (d/2) * np.log(detK) + (n/2) * np.log(np.linalg.det(omega)) + (1/2)*np.trace(A); 
    
    iter = 1
    Ns = X_s.shape[0]
    while ( Likelihood < -0.001 and iter<3 ):
        print("ITER", iter)
        Ns = Ns * 4.0 / 5
        X_s, Y_s = resample_numba( X_s, Y_s, n_samples = int(np.floor( Ns )) )
        K = get_covariance_numba(GA, PE, sigma, l, L, p, omega, noise, X_s)            
            
        Kinv = np.linalg.inv(K)
        omegainv = np.linalg.inv(omega)
        
        A = Kinv @ Y_s @ omegainv @ Y_s.T #  np.matmul(Kinv, np.matmul(self.Y_s, np.matmul(omegainv, np.transpose(self.Y_s))))
        detK = np.linalg.det(K)
        if detK < 0.0:
            detK = 0.01
        Likelihood = (n*d/2) * np.log(2*np.pi) + (d/2) * np.log(detK) + (n/2) * np.log(np.linalg.det(omega)) + (1/2)*np.trace(A)            
        if np.isnan(Likelihood):
            print("Error's error")
        iter = iter + 1
        # print("iter",iter)
    if np.isnan(Likelihood):
        print("ERROR ************************")
    
    # print("iter",iter)
    Kl, Ks, KL, Kp = get_dK_numba(GA, PE, sigma, l, L, p, omega, X_s) 
    dL_dl = (d/2)*np.trace( Kinv @ Kl ) + (1/2)*np.trace( -Kinv @ Kl @ A )
    dL_ds = (d/2)*np.trace( Kinv @ Ks ) + (1/2)*np.trace( -Kinv @ Ks @ A )
    dL_dL = (d/2)*np.trace( Kinv @ KL ) + (1/2)*np.trace( -Kinv @ KL @ A )
    dL_dp = (d/2)*np.trace( Kinv @ Kp ) + (1/2)*np.trace( -Kinv @ Kp @ A )
    dL_domega = (n/2) * omegainv.T - (1.0/2) * omegainv.T @ Y_s.T @ Kinv.T @ Y_s @ omegainv.T  #  (1/2)*np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(omegainv), np.transpose(self.Y_s)), np.transpose(Kinv)), self.Y_s), np.transpose(omegainv))
        
    if Likelihood<-0.01 and print_status:
        print(" *****************  WARN: Likelihood<0 *********************")
        Likelihood = 0.01
    
    return Likelihood, dL_dl, dL_ds, dL_domega, dL_dL, dL_dp, X_s, Y_s

def get_likelihood(GA, PE, sigma, l, L, p, omega, noise, X_s, Y_s, print_status = False):
    n = X_s.shape[0]
    d = Y_s.shape[1]
    
    K = get_covariance_numba(GA, PE, sigma, l, L, p, omega,  noise, X_s)               
    Kinv = np.linalg.inv(K)
    omegainv = np.linalg.inv(omega)        
    A = Kinv @ Y_s @ omegainv @ Y_s.T #  np.matmul(Kinv, np.matmul(self.Y_s, np.matmul(omegainv, np.transpose(self.Y_s))))
    
    detK = np.linalg.det(K)
    if detK < 0.0:
        detK = 0.0001
    Likelihood = (n*d/2) * np.log(2*np.pi) + (d/2) * np.log(detK) + (n/2) * np.log(np.linalg.det(omega)) + (1/2)*np.trace(A); 
    
    if np.isnan(Likelihood):
        print("ERROR ************************")
        return 1000.0
    
    return Likelihood

# @jit(nopython=True)
def train_numba(GA, PE, sigma, l, L, p, omega, noise, X_s, Y_s, n_samples = 100, max_iters = 100, print_status = False):
    
        # Define gradient descent parameters
        vals = []
        params_omega, params_sigma, params_l = [], [], []
        cur_o, cur_s, cur_l, cur_L, cur_p = omega, sigma, l, L, p
        temp_o, temp_s, temp_l, temp_L, temp_p = omega, sigma, l, L, p
        iters, alter_iter = 0, 2
        grad_max = 50.0
        omega_grad_max = 40.0
        rate = 0.005
        var = np.random.randint(3)
        
        Likelihood_previous = 10000
        
        while iters < max_iters:

            if (iters == 50):
                rate = 0.001
            if (iters == 100):
                rate = 0.0005

            # Get Gradients
            resample_numba( X_s, Y_s, n_samples = n_samples )
            Likelihood, dL_dl, dL_ds, dL_domega, dL_dL, dL_dp, X_temp, Y_temp = likelihood_gradients_numba(GA, PE, sigma, l, L, p, omega,  noise, X_s, Y_s, print_status = print_status)
            dL_domega = (dL_domega + np.transpose(dL_domega))/2
            dL_dl = np.clip(dL_dl, -grad_max, grad_max)
            dL_ds = np.clip(dL_ds, -grad_max, grad_max)
            dL_dL = np.clip(dL_dL, -grad_max, grad_max)
            dL_dp = np.clip(dL_dp, -grad_max, grad_max)
            if (np.amax(dL_domega) > omega_grad_max or np.amin(dL_domega) < omega_grad_max):
                max_val = max(np.amax(dL_domega), abs(np.amin(dL_domega)))
                dL_domega = dL_domega * (omega_grad_max / max_val)
            # print(f" dl: {dL_dl}, ds:{dL_ds}, domega:{dL_domega} ")
            # Gradient descent
            eps = 0.005
  
            
            
            iter_update = 0
            updated = False
            while (iter_update<1): 
                
                # Omega
                temp_o = cur_o - rate * dL_domega
                D, V = np.linalg.eig(cur_o)
                for i in range(len(D)):
                    if (D[i] <= eps):
                        D[i] = eps
                temp_o = V @ np.diag(D) @ np.linalg.inv(V) # np.matmul(np.matmul(V, np.diag(D)), np.linalg.inv(V))
                temp_o = (temp_o + np.transpose(temp_o))/2.0
                
                # Other parameters
                temp_l = np.clip(cur_l - rate * dL_dl, 0, None)
                temp_s = np.clip(cur_s - rate * dL_ds, 0, None)
                temp_L = np.clip(cur_L - rate * dL_dL, 0, None)
                temp_p = np.clip(cur_p - rate * dL_dp, 0, None)
                
                Likelihood_new = get_likelihood( GA, PE, temp_s, temp_l, temp_L, temp_p, temp_o,  noise, X_temp, Y_temp, print_status = print_status )
                print(f"Likelihhod:{Likelihood}, Likelihood new:{Likelihood_new}")
                # if (Likelihood_new < 0.0) and (Likelihood>0.0):
                #     rate = rate / 2.0
                # elif Likelihood_new < Likelihood:
                #     break
                # else:
                #     rate = rate / 2.0
                iter_update = iter_update + 1
                print("iter_update", iter_update)
                    
            # if Likelihood< 1.0 or iter_update >=9:
            #     # cannot improve??
            #     return sigma, l, L, p, omega
            
            cur_o, cur_l, cur_s, cur_L, cur_p = temp_o, temp_l, temp_s, temp_L, temp_p

            # Update parameters
            omega, sigma, l, L, p = cur_o, cur_s, cur_l, cur_L, cur_p
            # print("self.omega",self.omega)
            try: 
                np.linalg.inv(omega)
            except Exception as e:
                print("here ********************* OMEGA INV ERROR ********************************", e)
            iters = iters+1 #iteration count

            # value = self.log_likelihood()

            # Store and save updated parameters
            # params_omega.append(omega)
            # params_sigma.append(sigma)
            # params_l.append(l)
            # vals.append(Likelihood)
            print(f"sigma:{sigma}, l:{l}")
            # Save parameters and likelihoods
            if (iters % 1 == 0) and print_status:
                print(f"Iteration: {iters}, Likelihood for this dataset: {Likelihood}, grads: {dL_dl}, {dL_ds}, {dL_domega}, {dL_dL}, {dL_dp}")
                
        return sigma, l, L, p, omega

if (0):
    train_x = np.linspace(0,10,100).reshape(1,-1)

    train_y1 = np.cos(train_x)
    train_y2 = np.sin(train_x)
    train_y = np.append( train_y1, train_y2, axis=0 )

    omega = np.array([ [1.0, 0.0],[0.0, 1.0] ])
    sigma = 0.2 # put zero
    l = 2.0
    L = 0.1#5.0  # put zero
    p = 0.1#1.0
    noise = 0.01
    GA = 1
    PE = 0

    X_s, Y_s = resample_numba(train_x.T, train_y.T, n_samples = 100)
    cov_obs = get_covariance_numba( GA, PE, sigma, l, L, p, omega, noise, X_s  )
    K_inv = np.linalg.inv(cov_obs)
    
     
    # train_x = train_x[0,0:2].reshape(1,-1)
    # train_y = train_y[0:2,:]

    ys = []
    covs = []
    for i in range(train_x.shape[1]):
        # t0 = time.time()
        mu, cov = predict_numba( GA, PE, sigma, l, L, p, omega, noise, X_s, Y_s, K_inv, train_x[:,i].T )
        # print("Time taken ", time.time()-t0)
        # print(mu,cov)
        if ys == []:
            ys = np.copy(mu)
            covs = np.diagonal( cov ).reshape(1,-1)
        else:
            ys = np.append( ys, mu, axis=0 )
            covs = np.append( covs, np.diagonal( cov ).reshape(1,-1) , axis=0)



    factor = 5
   
    

    fig1, axis1 = plt.subplots(2,1)
    axis1[0].plot( train_x[0], train_y1[0], 'k--', label='True Value' )
    # axis1[0].plot( train_x[0], ys[:,0], 'r', label='Untrained Predicted Mean' )
    # axis1[0].fill_between( train_x[0], ys[:,0] - factor * covs[:,0], ys[:,0] + factor * covs[:,0], color="tab:orange", alpha=0.2 )

    axis1[1].plot( train_x[0], train_y2[0], 'k--', label='True Value' )
    # axis1[1].plot( train_x[0], ys[:,1], 'r', label='Untrained Predicted Mean' )
    # axis1[1].fill_between( train_x[0], ys[:,1] - factor * covs[:,1], ys[:,1] + factor * covs[:,1], color="tab:orange", alpha=0.2 )

    # plt.show()

    # sigma_, l_, L_, p_, omega_ = train_numba(GA, PE, sigma, l, L, p, omega, noise, X_s, Y_s, n_samples = 100, max_iters = 100, print_status = True)
    
    # cov_obs = get_covariance_numba( GA, PE, sigma_, l_, L_, p_, omega_, noise, X_s  )
    # K_inv = np.linalg.inv(cov_obs)

    # ys = []
    # covs = []
    # for i in range(train_x.shape[1]):
    #     # t0 = time.time()
    #     mu, cov = predict_numba( GA, PE, sigma_, l_, L_, p_, omega_, noise, X_s, Y_s, K_inv, train_x[:,i].T )
    #     # print("Time taken ", time.time()-t0)
    #     # print(mu,cov)
    #     if ys == []:
    #         ys = np.copy(mu)
    #         covs = np.diagonal( cov ).reshape(1,-1)
    #     else:
    #         ys = np.append( ys, mu, axis=0 )
    #         covs = np.append( covs, np.diagonal( cov ).reshape(1,-1) , axis=0)
            
    # axis1[0].plot( train_x[0], ys[:,0], 'g', label='Trained Predicted Mean' )
    # axis1[0].fill_between( train_x[0], ys[:,0] - factor * covs[:,0], ys[:,0] + factor * covs[:,0], color="tab:blue", alpha=0.2 )

    # axis1[1].plot( train_x[0], ys[:,1], 'g', label='Trained Predicted Mean' )
    # axis1[1].fill_between( train_x[0], ys[:,1] - factor * covs[:,1], ys[:,1] + factor * covs[:,1], color="tab:blue", alpha=0.2 )
            
    # plt.show()
            
    kern = ConstantKernel(constant_value=2.0, constant_value_bounds=(1.0, 1e12)) * RBF(length_scale=2.0, length_scale_bounds=(1.0, 1e12))
    print("bounds", kern.bounds)
    gp_sklearn1 = GaussianProcessRegressor(kernel=kern, alpha = noise, n_restarts_optimizer=10)
    gp_sklearn2 = GaussianProcessRegressor(kernel=kern, alpha = noise, n_restarts_optimizer=10)

    gp_sklearn1.fit(X_s,Y_s[:,0])
    gp_sklearn2.fit(X_s,Y_s[:,1])
            
    ys1 = []
    covs1 = []
    for i in range(train_x.shape[1]):
        # t0 = time.time()
        mu1, std1 = gp_sklearn1.predict( train_x[:,i].reshape(1,-1)  , return_std=True )
        mu2, std2 = gp_sklearn2.predict( train_x[:,i].reshape(1,-1)  , return_std=True )
        mu = np.append( mu1.reshape(-1,1), mu2.reshape(-1,1), axis = 1 )
        cov = np.append( std1.reshape(-1,1), std2.reshape(-1,1), axis = 1 )
        # print("Time taken ", time.time()-t0)
        # print(mu,cov)
        if i==0:#ys == []:
            ys1 = np.copy(mu)
            covs1 = np.copy( cov )
        else:
            ys1 = np.append( ys1, mu, axis=0 )
            covs1 = np.append( covs1, cov , axis=0)
            
    axis1[0].plot( train_x[0], ys1[:,0], 'r', label='Trained Predicted Mean' )
    axis1[0].fill_between( train_x[0], ys1[:,0] - factor * covs1[:,0], ys1[:,0] + factor * covs1[:,0], color="tab:orange", alpha=0.2 )

    axis1[1].plot( train_x[0], ys1[:,1], 'r', label='Trained Predicted Mean' )
    axis1[1].fill_between( train_x[0], ys1[:,1] - factor * covs1[:,1], ys1[:,1] + factor * covs1[:,1], color="tab:orange", alpha=0.2 )
            
    s1, l1 = np.exp( gp_sklearn1.kernel_.theta )
    s2, l2 = np.exp( gp_sklearn2.kernel_.theta )
    print(f"s1:{s1}, s2:{s2}, l1:{l1}, l2:{l2}")

    
    # print(f"ys1:{ys1}, cos1:{covs1}")     
    
    omega = np.array([[1.0]])
    
    cov_obs1 = get_covariance_numba( GA, PE, s1, l1, L, p, omega, noise, gp_sklearn1.X_train_  )
    cov_obs2 = get_covariance_numba( GA, PE, s2, l2, L, p, omega, noise, gp_sklearn2.X_train_  )
    
    K_inv1 = np.linalg.inv(cov_obs1)
    K_inv2 = np.linalg.inv(cov_obs2)
    
    ys = []
    covs = []
    for i in range(train_x.shape[1]):
        # t0 = time.time()
        mu1, cov1 = predict_numba( GA, PE, s1, l1, L, p, omega, noise, gp_sklearn1.X_train_, Y_s[:,0], K_inv1, train_x[:,i].T ) 
        mu2, cov2 = predict_numba( GA, PE, s2, l2, L, p, omega, noise, gp_sklearn2.X_train_, Y_s[:,1], K_inv2, train_x[:,i].T ) 
        mu = np.append( mu1.reshape(-1,1), mu2.reshape(-1,1), axis = 1 )
        cov = np.append( cov1.reshape(-1,1), cov2.reshape(-1,1), axis = 1 )
        # print("Time taken ", time.time()-t0)
        # print(mu,cov)
        if i==0:#ys == []:
            ys = np.copy(mu)
            covs = np.copy( cov )
        else:
            ys = np.append( ys, mu, axis=0 )
            covs = np.append( covs, cov , axis=0)    
            
    # print(f"ys:{ys}, cos:{covs}")     
            
    axis1[0].plot( train_x[0], ys[:,0], 'g', label='Trained Predicted Mean' )
    axis1[0].fill_between( train_x[0], ys[:,0] - factor * covs[:,0], ys[:,0] + factor * covs[:,0], color="tab:blue", alpha=0.2 )

    axis1[1].plot( train_x[0], ys[:,1], 'g', label='Trained Predicted Mean' )
    axis1[1].fill_between( train_x[0], ys[:,1] - factor * covs[:,1], ys[:,1] + factor * covs[:,1], color="tab:blue", alpha=0.2 )
            
    plt.show()