import numpy as np
import cvxpy as cp

def connectivity_undirected_laplacian(robots, max_dist):
    
    # Adjacency Matrix
    A = np.zeros( ( len(robots), len(robots) ) )    
    for i in range( len(robots) ):
        for j in range( i, len(robots) ):
            if np.linalg.norm( robots[i].X[0:2] - robots[j].X[0:2] ) < max_dist:
            # or any other criteria
                A[i, j] = 1
                A[j, i] = 1
                
    # Degree matrix
    D = np.diag( np.sum( A, axis = 1 ) )
    
    # Laplacian Matrix
    L = D - A
    return L

def weighted_connectivity_undirected_laplacian(robots, max_dist = 1.0):
    
    # thresholds
    rho =  1.0 #1.0 #0.5
    gamma = 0.5
    
    # Adjacency Matrix
    A = np.zeros( ( len(robots), len(robots) ) )
    
    for i in range( len(robots) ):
        robots[i].dL_dx = np.zeros( ( len(robots), len(robots), np.shape(robots[i].X)[0] ) )
        robots[i].dA_dx = np.zeros( ( len(robots), len(robots), np.shape(robots[i].X)[0] ) )
    
    for i in range( len(robots) ):
        
        for j in range( i+1, len(robots) ):
            
            # weight
            dist = np.linalg.norm( robots[i].X[0:2] - robots[j].X[0:2] )
            
            # weight gradient
            d_dist_dxi = 1.0/dist * (robots[i].X[0:2] - robots[j].X[0:2] ).reshape(1,-1)
            d_dist_dxj = - 1.0/dist * (robots[i].X[0:2] - robots[j].X[0:2] ).reshape(1,-1)
            
            # derivative w.r.t state
            der_i = np.array([0,0]).reshape(1,-1)
            der_j = np.array([0,0]).reshape(1,-1)
                
            if dist <= rho:
                A[i , j] = 1.0                
            elif dist >= max_dist:
                A[i, j] = 0.0
            else:
                A[i, j] = np.exp( -gamma * (dist-rho) / (max_dist-rho)  )
                der_i = A[i , j] * ( -gamma/(max_dist-rho) * d_dist_dxi )
                der_j = A[i , j] * ( -gamma/(max_dist-rho) * d_dist_dxj )
            # or any other criteria
            A[j, i] = A[i, j]
            
            # i's Adjacency derivatives
            robots[i].dA_dx[i,j,:] = der_i
            robots[i].dA_dx[j,i,:] = der_i
            
            # j's Adjacency derivatives
            robots[j].dA_dx[i,j,:] = der_j
            robots[j].dA_dx[j,i,:] = der_j
            
            # Laplacian Derivatives
            robots[i].dL_dx[i,j,:] = - robots[i].dA_dx[i,j,:]
            robots[i].dL_dx[j,i,:] = - robots[i].dA_dx[j,i,:]
            robots[i].dL_dx[i,i,:] = robots[i].dL_dx[i,i,:] + robots[i].dA_dx[i,j,:]
            robots[i].dL_dx[j,j,:] = robots[i].dL_dx[j,j,:] + robots[i].dA_dx[j,i,:]
            
            robots[j].dL_dx[i,j,:] = - robots[j].dA_dx[i,j,:]
            robots[j].dL_dx[j,i,:] = - robots[j].dA_dx[j,i,:]
            robots[j].dL_dx[i,i,:] = robots[j].dL_dx[i,i,:] + robots[j].dA_dx[i,j,:]
            robots[j].dL_dx[j,j,:] = robots[j].dL_dx[j,j,:] + robots[j].dA_dx[j,i,:]
            
    # Degree matrix
    D = np.diag( np.sum( A, axis = 1 ) )
    
    # Laplacian Matrix
    L = D - A
    return L

def leader_weighted_connectivity_undirected_laplacian(robots, max_dist = 1.0):
    
    # thresholds
    rho =  1.0 #1.0 #0.5
    gamma = 0.5
    
    # Adjacency Matrix
    A = np.zeros( ( len(robots), len(robots) ) )
    
    for i in range( len(robots) ):
        robots[i].dL_dx = np.zeros( ( len(robots), len(robots), np.shape(robots[i].X)[0] ) )
        robots[i].dA_dx = np.zeros( ( len(robots), len(robots), np.shape(robots[i].X)[0] ) )
    
    for i in range( len(robots) ):
        
        dist_leader = 0
        for j in range( i+1, len(robots) ):
            
            # weight
            dist = np.linalg.norm( robots[i].X[0:2] - robots[j].X[0:2] )
            
            # weight gradient
            d_dist_dxi = 1.0/dist * (robots[i].X[0:2] - robots[j].X[0:2] ).reshape(1,-1)
            d_dist_dxj = - 1.0/dist * (robots[i].X[0:2] - robots[j].X[0:2] ).reshape(1,-1)
            
            # derivative w.r.t state
            der_i = np.array([0,0]).reshape(1,-1)
            der_j = np.array([0,0]).reshape(1,-1)
                
            if dist <= rho:
                A[i , j] = 1.0                
            elif dist >= max_dist:
                A[i, j] = 0.0
            else:
                A[i, j] = np.exp( -gamma * (dist-rho) / (max_dist-rho)  )
                der_i = A[i , j] * ( -gamma/(max_dist-rho) * d_dist_dxi )
                der_j = A[i , j] * ( -gamma/(max_dist-rho) * d_dist_dxj )
            
            # Add leader connection weight to this and see what happens!    
            if i>0: # 1,2   
                # i: robot, j:leader
                der_i = der_i * A[i, 0] * A[j, 0] + A[i, j] * robots[i].dA_dx[i,0,:] * A[j, 0]
                der_j = der_j * A[i, 0] * A[j, 0] + A[i, j] * A[i, 0] * robots[j].dA_dx[j,0,:]
                A[i, j] = A[i, j] * A[i, 0] * A[j, 0]
                
            # or any other criteria
            A[j, i] = A[i, j]
            
            # i's Adjacency derivatives
            robots[i].dA_dx[i,j,:] = der_i
            robots[i].dA_dx[j,i,:] = der_i
            
            # j's Adjacency derivatives
            robots[j].dA_dx[i,j,:] = der_j
            robots[j].dA_dx[j,i,:] = der_j
            
            # Laplacian Derivatives
            robots[i].dL_dx[i,j,:] = - robots[i].dA_dx[i,j,:]
            robots[i].dL_dx[j,i,:] = - robots[i].dA_dx[j,i,:]
            robots[i].dL_dx[i,i,:] = robots[i].dL_dx[i,i,:] + robots[i].dA_dx[i,j,:]
            robots[i].dL_dx[j,j,:] = robots[i].dL_dx[j,j,:] + robots[i].dA_dx[j,i,:]
            
            robots[j].dL_dx[i,j,:] = - robots[j].dA_dx[i,j,:]
            robots[j].dL_dx[j,i,:] = - robots[j].dA_dx[j,i,:]
            robots[j].dL_dx[i,i,:] = robots[j].dL_dx[i,i,:] + robots[j].dA_dx[i,j,:]
            robots[j].dL_dx[j,j,:] = robots[j].dL_dx[j,j,:] + robots[j].dA_dx[j,i,:]
            
    # Degree matrix
    D = np.diag( np.sum( A, axis = 1 ) )
    
    # Laplacian Matrix
    L = D - A
    return L

def modify_weighted_connectivity_undirected_laplacian(robots, L, j, k):
    # remove k from j's neighbor: has to be directed here
    L[j, k] = 0  # global version
    robots[j].dL_dx_copy[i,j,:] = 0  # local version
    robots[j].dL_dx_copy[j,i,:] = 0
    return L

def mofify_lambda2_dx( robots, L, Lambda2, V2 ):
    dLambda2_dL = V2 @ V2.T / ( V2.T @ V2 )
    
    for i in range( len(robots) ):
        robots[i].lambda2_dx = np.zeros( (1,robots[i].X.shape[0]) )
        for j in range( np.shape( robots[i].X )[0] ):
            robots[i].lambda2_dx[0,j] =  np.trace( dLambda2_dL.T @ robots[i].dL_dx_copy[:,:,j]  )

def lambda2_dx( robots, L, Lambda2, V2 ):
    dLambda2_dL = V2 @ V2.T / ( V2.T @ V2 )
    
    for i in range( len(robots) ):
        robots[i].lambda2_dx = np.zeros( (1,robots[i].X.shape[0]) )
        for j in range( np.shape( robots[i].X )[0] ):
            robots[i].lambda2_dx[0,j] =  np.trace( dLambda2_dL.T @ robots[i].dL_dx[:,:,j]  )
            # print(f" dl_dx:{ robots[i].lambda2_dx[0,j] } ")
    
  
# Eigenvalue and Eigenvectors of laplacian Matrix: 
def laplacian_eigen( L ):
   Lambda, V = np.linalg.eig(L)  # eigenvalues, right eigenvectorsb
   eigenvalue_order = np.argsort(Lambda)
   Lambda = Lambda[eigenvalue_order]
   V = V[:, eigenvalue_order]
   return Lambda, V

def directed_milp_r_robustness( L ):
    n = np.shape(L)[0]
    b = cp.Variable((2*n,1) , integer=True)#boolean = True )
    t = cp.Variable()
    obj = cp.Minimize(t)
    L2 = np.kron( np.eye(2), L )
    
    const = []        
    const += [t >= 0]
    const += [ L2 @ b <= t * np.ones((2*n,1)) ]
    const += [ b >= np.zeros((2*n,1)), b <= np.ones((2*n,1)) ] # binary constraint
    const += [ np.append( np.eye(n), np.eye(n), axis=1 ) @ b <= 1  ]     
    const += [ np.append( np.ones((1,n)), np.zeros((1,n)), axis=1) @ b >= 1 ]   
    const += [ np.append( np.ones((1,n)), np.zeros((1,n)), axis=1) @ b <= n-1 ]
    const += [ np.append( np.zeros((1,n)), np.ones((1,n)), axis=1) @ b >= 1 ]
    const += [ np.append( np.zeros((1,n)), np.ones((1,n)), axis=1) @ b <= n-1 ]
    
    prob = cp.Problem( obj, const )
    prob.solve(solver=cp.GUROBI, reoptimize=True)
    # print(f"r: {t.value}")
    
    return t.value

def directed_milp_rs_robustness( L, r ):
    n = np.shape(L)[0]
    s = cp.Variable()
    b1 = cp.Variable((n,1) , integer=True)#, boolean = True)
    b2 = cp.Variable((n,1) , integer=True)#, boolean = True)
    y1 = cp.Variable((n,1) , integer=True)#, boolean = True)
    y2 = cp.Variable((n,1) , integer=True)#, boolean = True)
    ones = np.ones((n,1))
    
    obj = cp.Minimize( s )
    
    const = []
    const += [ s >= 1, s<=(n+1) ]
    const += [ ones.T @ y1 <= ones.T @ b1 - 1 ]
    const += [ ones.T @ y2 <= ones.T @ b2 - 1 ]
    const += [ ones.T @ y1 + ones.T @ y2 <= s - 1 ]
    const += [ L @ b1 - n * y1 <= (r-1) * ones ]
    const += [ L @ b2 - n * y2 <= (r-1) * ones ]
    const += [ b1 + b2 <= 1 ]
    const += [ 1 <= ones.T @ b1, ones.T @ b1 <= n-1 ]
    const += [ 1<= ones.T @ b2, ones.T @ b2 <= n-1 ]
    
    prob = cp.Problem( obj, const )
    prob.solve(solver=cp.GUROBI, reoptimize=True)
    # print(f"s status: {prob.status}, s:{s.value}")
    return s.value
    
L = np.array( [ [1, -1, 0],
                [-1, 2, -1],
                [-1, -1, 2]] ) 
# L = np.array( [ [3, -1, -1, -1],
#                 [-1, 3, -1, -1],
#                 [-1, -1, 3, -1],
#                 [-1, -1, -1, 3]] ) 
# L = np.array([[1.0]])   
# L = np.array( [ [2.0, 2.0, 0.0],
#                 [2.0, 3.3, 1.1],
#                 [0.0, 0.9, 0.9]] )
# L = np.array( [ [2.0, 1.0, 1.0],
#                 [1.0, 2.0, 1.0],
#                 [1.0, 1.0, 2.0]] )
# L = np.array( [ [1.5, 1.0, 0.5],
#                 [1.0, 2.0, 1.0],
#                 [0.5, 1.0, 1.5]] ) # 0.5
# L = np.array( [ [2.0, 1.5, 0.5],
#                 [1.5, 2.5, 1.0],
#                 [0.5, 1.0, 1.5]] ) # 1.5
# L = np.array( [ [2.0, 1.5, 0.5],
#                 [1.5, 1.51, 0.01],
#                 [0.0, 0.2, 0.2]] ) # 0.5
# L = np.array( [ [1.7, 1.5, 0.2],
#                 [0.4, 0.4, 0.0],
#                 [0.0, 0.2, 0.2]] )
# L = np.array( [ [0.0, 1.0, 0.0],
#                 [0.0, 0.4, 0.0],
#                 [0.0, 0.2, 0.2]] )
r = directed_milp_r_robustness( L )
s = directed_milp_rs_robustness( L, r )



## Compute r-robustness



# Mixed Integer for finding r robustness: exact solution: prof wants me to do it    


#