import torch
import numpy as np
import time
from robot_models.UnicycleJIT import *
from utils.ut_utilsJIT import *
from utils.sqrtm import sqrtm

y = unicycle_f_torch_jit(torch.tensor([1,2,0.4]))
y = unicycle_f_torch_jit(torch.tensor([1,2,0.4]))
t0 = time.time()
y = unicycle_f_torch_jit(torch.tensor([1,2,0.4]))
print(f"time taken: {time.time()-t0}")

x = torch.tensor([1,2,np.pi/2]).reshape(-1,1)
y = torch.tensor([1, 3, np.pi/2]).reshape(-1,1)
unicycle_SI2D_barrier_torch_jit(x,y)
unicycle_SI2D_barrier_torch_jit(x,y)
t0 = time.time()
unicycle_SI2D_barrier_torch_jit(x,y)
print(f"time taken barrier: {time.time()-t0}")

points = torch.tensor( [ [ 1,2 ], [3, 4] ], dtype=torch.float, requires_grad=True )
weights = torch.tensor( [0.5, 0.5] )
mu = get_mean_JIT(points, weights)
print("mean",mu)
mu, cov = get_mean_cov_JIT(points, weights)
print(f"mean:{mu}, cov:{cov}")
mu.sum().backward()
print("grad", points.grad)
root_term = []
if torch.linalg.det( cov ) < 0.01:
    root_term = cov
else:
    root_term = sqrtm((n+k)*cov)
generate_sigma_points_JIT(mu, cov, root_term)

sigma_point_scale_up3_JIT( points, points )