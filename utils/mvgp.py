import numpy as np
import random
import torch

class MVGP:
    """
    General purpose Gaussian process model
    :param X: input observations: size N x n. N: no of observations, n: dimension of a single input vector
    :param Y: output observations: size N x m. N: no of observations, m: dimension of a single output vector
    :param kernel: a GP kernel, defaults to squared exponential
    :param likelihood: a GPy likelihood
    :param inference_method: The :class:`~GPy.inference.latent_function_inference.LatentFunctionInference` inference method to use for this GP
    :rtype: model object
    :param Norm normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is True, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    .. Note:: Multiple independent outputs are allowed using columns of Y
    """
    def __init__(self, X = [], Y = [], kernel='SE', omega=None, l=None, sigma=1.0, noise=None, horizon=20):
        self.X = X
        self.Y = Y
        self.X_s = X
        self.Y_s = Y
        self.kernel = kernel
        self.omega = omega
        self.l = l
        self.sigma = sigma
        self.noise = noise
        self.K = None                                   # Train GP
        self.K_obs = np.empty((horizon, horizon))       # Observation GP
        self.K_star = np.empty(horizon)
        self.N_data = 0
        self.horizon = horizon
        self.X_obs = []
        self.Y_obs = []
        self.count = -1
        
        self.sigma_torch = torch.tensor(self.sigma, dtype=torch.float)
        self.K_obs_torch = torch.zeros( (horizon,horizon), dtype=torch.float )
        self.K_star_torch = torch.zeros(horizon, dtype=torch.float)
        self.Y_obs_torch = []
        self.X_obs_torch = []
        
        self.kstars = []

    def load_parameters(self, file_name):
        # open a file, where you stored the pickled data
        file = open(file_name, 'rb')
        data = pickle.load(file)
        file.close()
        self.omega = data['omega']
        self.sigma = data['sigma']
        self.l = data['l']

    # Set/update input data for model
    def set_XY(self, X, Y):
        """
        Set the input data of the model
        :param X: input observations
        :type X: np.ndarray
        """
        self.X = X
        self.Y = Y
            
    def add_data(self, x, y):
        self.X.append(np.copy(x))
        self.Y.append(np.copy(y))
        if (len(self.X) != len(self.Y)):
            print("ERROR: Input/output data dimensions don't match")
        if (len(self.X) > self.horizon):
            self.X.pop(0)
            self.Y.pop(0)
        self.N_data = len(self.X)
        self.count += 1
        if (self.count >= self.horizon):
            self.count = 0
            
    # select a subset for predictions
    def resample_obs( self, n_samples=80, start_index = 0 ):
        N = self.X.shape[0]
        idx = random.sample(range(0, N), min(n_samples, N))
        self.X_obs, self.Y_obs = self.X[idx,:], self.Y[idx,:]
        self.N_data = self.X_obs.shape[0]
        return self.X_obs, self.Y_obs

        # Evaluate kernel (squared exponential)
    def evaluate_kernel(self, x1, x2):
        diff = np.linalg.norm(x1 - x2)
        return self.sigma**2 * np.exp(-diff**2 / (2*self.l**2))
    
    # Get K*
    def get_X_cov(self, Xnew):
        N = self.N_data
        for i in range(N):
            self.K_star[i] = self.evaluate_kernel(self.X_obs[i,:], Xnew)
        return self.K_star[0:N]

    # Get covariance matrix given current dataset
    def get_obs_covariance(self):
        N = self.N_data
        K = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.evaluate_kernel(self.X_obs[i,:], self.X_obs[j,:])
                if (i == j):
                    K[i, i] = val + self.noise
                else:
                    K[i, j] = val
                    K[j, i] = val
        self.K_obs = K
        return K

    # Update covariance matrix given new data (run after add_data)
    def update_obs_covariance(self):
        N = self.N_data
        x = self.X_obs[-1,:]
        for i in range(N):
            val = self.evaluate_kernel(x, self.X_obs[i,:])
            if (i == N-1):
                self.K_obs[N-1, N-1] = val + self.noise
            else:
                self.K_obs[i, N-1] = val
                self.K_obs[N-1, i] = val
        return self.K_obs[0:N, 0:N]

    # Predict function at new point Xnew
    def predict(self, Xnew):
        N = self.N_data
        # K = self.get_covariance()
        K_inv = np.linalg.inv(self.K_obs[0:N,0:N])
        k_star = self.get_X_cov(Xnew)
        self.kstars.append(k_star)
        mean = (K_inv @ k_star).T @ self.Y_obs #np.matmul(np.transpose(np.matmul(K_inv, k_star)), self.Y_obs[0:N])
        Sigma = self.evaluate_kernel(Xnew, Xnew) + self.noise -  (K_inv @ k_star).T @ k_star
        
        # mean = np.matmul(np.transpose(np.matmul(K_inv, k_star)), self.Y_obs[0:N])
        # Sigma = self.evaluate_kernel(Xnew, Xnew) + self.noise - np.matmul(np.transpose(np.matmul(K_inv, k_star)), k_star)
        cov = np.kron(Sigma, self.omega)
        return mean.reshape(1,-1), cov
    
    # for prediction in torch
    def initialize_torch(self):
        self.K_obs_torch = torch.tensor(self.K_obs, dtype = torch.float )
        self.Y_obs_torch = torch.tensor( self.Y_obs, dtype = torch.float )
        self.X_obs_torch = torch.tensor( self.X_obs, dtype=torch.float )
        self.K_star_torch = torch.zeros( (self.N_data,1), dtype=torch.float )
        self.omega_torch = torch.tensor( self.omega, dtype=torch.float )
        self.l_torch = torch.tensor( self.l, dtype=torch.float )
        self.sigma_torch = torch.tensor( self.sigma, dtype=torch.float )
        self.K_inv_torch = torch.inverse(self.K_obs_torch)
        self.kstars = []
        
    def evaluate_kernel_torch(self, x1, x2):
        diff = torch.norm(x1 - x2)
        return self.sigma_torch**2 * torch.exp(-diff**2 / (2*self.l_torch**2))
        
    def get_X_cov_torch(self,Xnew):
        N = self.N_data
        for i in range(N):
            self.K_star_torch[i,0] = self.evaluate_kernel_torch(self.X_obs_torch[i,:], Xnew)
        return self.K_star_torch
    
    def predict_torch(self, Xnew):
        N = self.N_data
        # K = self.get_covariance()
        
        # if N==0:
        #     return torch.zeros(2), torch.eye(2)
                
        # K_inv = torch.inverse(self.K_obs_torch[0:N,0:N])
        k_star = self.get_X_cov_torch(Xnew)
        mean =  (self.K_inv_torch @ k_star).T @ self.Y_obs_torch
        
        # print("**************** called *****************")
        Sigma = self.evaluate_kernel_torch(Xnew, Xnew) + self.noise - (self.K_inv_torch @ k_star).T @ torch.clone(k_star)
        cov = torch.kron(Sigma, self.omega_torch)
        return mean, cov
    
    # For TRAINING
    
     # Sample subset of data for gradient computation
    def resample(self, n_samples=80, start_index = 0):
        N = self.X.shape[0]
        idx = random.sample(range(0, N), min(n_samples, N))
        self.X_s, self.Y_s = self.X[idx], self.Y[idx]
        return self.X_s, self.Y_s

    # Evaluate derivative of kernel (w.r.t. length scale)
    def dk_dl(self, x1, x2):
        diff = np.linalg.norm(x1 - x2)
        return self.sigma**2 * np.exp(-diff**2 / (2*self.l**2)) * (diff**2 / (self.l**3))

    # Evaluate derivative of kernel (w.r.t. sigma)
    def dk_ds(self, x1, x2):
        diff = np.linalg.norm(x1 - x2)
        return 2*self.sigma * np.exp(-diff**2 / (2*self.l**2))

    # Get covariance matrix given current dataset
    def get_covariance(self):
        N = len(self.X_s)
        K = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.evaluate_kernel(self.X_s[i,:], self.X_s[j,:])
                if (i == j):
                    K[i, i] = val + self.noise
                else:
                    K[i, j] = val
                    K[j, i] = val
        self.K = K
        return K

    # Get derivative of covariance matrix (w.r.t. length scale and sigma)
    def get_dK(self):
        N = len(self.X_s)
        Kl = np.empty((N, N))
        Ks = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                val_l = self.dk_dl(self.X_s[i,:], self.X_s[j,:])
                val_s = self.dk_ds(self.X_s[i,:], self.X_s[j,:])
                if (i == j):
                    Kl[i, i] = val_l
                    Ks[i, i] = val_s
                else:
                    Kl[i, j] = val_l
                    Kl[j, i] = val_l
                    Ks[i, j] = val_s
                    Ks[j, i] = val_s
        return Kl, Ks

    # Get gradient of negative log likelihood (w.r.t. length scale, sigma, omega)
    def likelihood_gradients(self, print_status = False):
        n = self.X_s.shape[0]
        d = self.Y_s.shape[1]
        
        K = self.get_covariance()               
        Kinv = np.linalg.inv(K)
        omegainv = np.linalg.inv(self.omega)        
        A = Kinv @ self.Y_s @ omegainv @ self.Y_s.T #  np.matmul(Kinv, np.matmul(self.Y_s, np.matmul(omegainv, np.transpose(self.Y_s))))
        L = (n*d/2) * np.log(2*np.pi) + (d/2) * np.log(np.linalg.det(K)) + (n/2) * np.log(np.linalg.det(self.omega)) + (1/2)*np.trace(A); 
        
        iter = 1
        Ns = self.X_s.shape[0]
        while ( L < -0.001 and iter<5 ):
            Ns = Ns * 4.0 / 5
            self.resample( int(np.floor( Ns )) )
            K = self.get_covariance()            
                
            Kinv = np.linalg.inv(K)
            omegainv = np.linalg.inv(self.omega)
            
            A = Kinv @ self.Y_s @ omegainv @ self.Y_s.T #  np.matmul(Kinv, np.matmul(self.Y_s, np.matmul(omegainv, np.transpose(self.Y_s))))
            L = (n*d/2) * np.log(2*np.pi) + (d/2) * np.log(np.linalg.det(K)) + (n/2) * np.log(np.linalg.det(self.omega)) + (1/2)*np.trace(A);
            iter = iter + 1
            
        # print("iter",iter)
        Kl, Ks = self.get_dK() 
        dL_dl = (d/2)*np.trace( Kinv @ Kl ) + (1/2)*np.trace( -Kinv @ Kl @ A )
        dL_ds = (d/2)*np.trace( Kinv @ Ks ) + (1/2)*np.trace( -Kinv @ Ks @ A )
        dL_domega = (n/2) * omegainv.T - (1.0/2) * omegainv.T @ self.Y_s.T @ Kinv.T @ self.Y_s @ omegainv.T  #  (1/2)*np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(omegainv), np.transpose(self.Y_s)), np.transpose(Kinv)), self.Y_s), np.transpose(omegainv))
            
        if L<-0.01 and print_status:
            print(" *****************  WARN: L<0 *********************")
            L = 0.01
        
        self.resample(n_samples = n)       
        
        return dL_dl, dL_ds, dL_domega

    # Compute negative log likelihood
    def log_likelihood(self):
       
        n = self.X_s.shape[0]
        d = self.Y_s.shape[1]
        
        K = self.get_covariance()               
        Kinv = np.linalg.inv(K)
        omegainv = np.linalg.inv(self.omega)        
        A = Kinv @ self.Y_s @ omegainv @ self.Y_s.T #  np.matmul(Kinv, np.matmul(self.Y_s, np.matmul(omegainv, np.transpose(self.Y_s))))
        L = (n*d/2) * np.log(2*np.pi) + (d/2) * np.log(np.linalg.det(K)) + (n/2) * np.log(np.linalg.det(self.omega)) + (1/2)*np.trace(A);        
        
        return L
    

    def train(self, n_samples = 100, max_iters = 100, print_status = False):

        # Define gradient descent parameters
        vals = []
        params_omega, params_sigma, params_l = [], [], []
        cur_o, cur_s, cur_l = self.omega, self.sigma, self.l 
        iters, alter_iter = 0, 2
        grad_max = 50.0
        omega_grad_max = 40.0
        rate = 0.005
        var = np.random.randint(3)
        
        while iters < max_iters:
            prev_o, prev_s, prev_l = self.omega, self.sigma, self.l
            
            if (iters == 5000):
                rate = 0.0003
            if (iters == 10000):
                rate = 0.0002

            # Get Gradients
            self.resample( n_samples = n_samples )
            dL_dl, dL_ds, dL_domega = self.likelihood_gradients(print_status = print_status)
            dL_domega = (dL_domega + np.transpose(dL_domega))/2
            dL_dl = np.clip(dL_dl, -grad_max, grad_max)
            dL_ds = np.clip(dL_ds, -grad_max, grad_max)
            if (np.amax(dL_domega) > omega_grad_max or np.amin(dL_domega) < omega_grad_max):
                max_val = max(np.amax(dL_domega), abs(np.amin(dL_domega)))
                dL_domega = dL_domega * (omega_grad_max / max_val)
            # print(f" dl: {dL_dl}, ds:{dL_ds}, domega:{dL_domega} ")
            # Gradient descent
            eps = 0.0005
            if (var == 0):
                cur_o = cur_o - rate * dL_domega
                D, V = np.linalg.eig(cur_o)
                for i in range(len(D)):
                    if (D[i] <= eps):
                        D[i] = eps
                cur_o = V @ np.diag(D) @ np.linalg.inv(V) # np.matmul(np.matmul(V, np.diag(D)), np.linalg.inv(V))
            elif (var == 1):
                cur_l = cur_l - rate * dL_dl
            elif (var == 2):
                cur_s = cur_s - rate * dL_ds
            else:
                print("Error in parameter update")

            # sigma cannot be negative so constrain it
            if cur_s < 0:
                cur_s = 0

            # Update parameters
            self.omega, self.sigma, self.l = cur_o, cur_s, cur_l
            self.omega = (self.omega + np.transpose(self.omega))/2
            # print("self.omega",self.omega)
            try: 
                np.linalg.inv(self.omega)
            except Exception as e:
                print("here", e)
            iters = iters+1 #iteration count
            value = self.log_likelihood()

            # Store and save updated parameters
            params_omega.append(self.omega)
            params_sigma.append(self.sigma)
            params_l.append(self.l)
            vals.append(value)

            if (iters % alter_iter == 0):
                if (var < 2):
                    var += 1
                elif (var == 2):
                    var = 0
                else:
                    print("Error in setting variable to update.")

            # Save parameters and likelihoods
            if (iters % 10 == 0) and print_status:
                print(f"Iteration: {iters}, Likelihood for this dataset: {value}, grads: {dL_dl}, {dL_ds}, {dL_domega}")
