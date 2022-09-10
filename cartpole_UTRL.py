



# Nonlinear RBF network
def policy(param_w, param_mu, param_Sigma, X):
    n = 4 # dim of state
    m = 1 # dim of input
    N = 50 # number of basis functions
    mu = params[]
    for i in range(N):
        diff = X - 
        phi_i = torch.exp( -0.5 *  )