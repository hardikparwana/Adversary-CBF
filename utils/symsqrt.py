import torch
import scipy
from scipy.linalg import sqrtm
 
def symsqrt(a, cond=None, return_rank=False):
    """Computes the symmetric square root of a positive definite matrix"""

    s, u = torch.symeig(a, eigenvectors=True)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}
    print(f"s:{s}, u:{u}")

    if cond in [None, -1]:
        cond = cond_dict[a.dtype]

    above_cutoff = (abs(s) > cond * torch.max(abs(s)))
    print("above cutoof", above_cutoff)

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]
    
    above_cutoff = (abs(s) > cond * torch.max(abs(s)))
    print("above cutoof", above_cutoff)

    psigma_diag = torch.sqrt(s)

    B = u @ torch.diag(psigma_diag) @ u.t()
    if return_rank:
        return B, len(psigma_diag)
    else:
        return B

# def symsqrt_test():
#     def randomly_rotate(X):
#         """Randomly rotate d,n data matrix X"""
#         d, n = X.shape
#         z = torch.randn((d, d), dtype=X.dtype)
#         q, r = torch.qr(z)
#         d = torch.diag(r)
#         ph = d / abs(d)
#         rot_mat = q * ph
#         return rot_mat @ X

#     n = 20
#     d = 10
#     X = torch.randn((d, n))

#     # embed in a larger space
#     X = torch.cat([X, torch.zeros_like(X)])
#     X = randomly_rotate(X)
#     cov = X @ X.t()
#     sqrt, rank = symsqrt(cov, return_rank=True)
#     assert rank == d
#     assert torch.allclose(sqrt @ sqrt, cov, atol=1e-5)

# x = torch.tensor([ [3.0, 0.75],[0.75, 2.0] ], requires_grad=True)
x = torch.tensor([ [3.0, 0.0],[0.0, 3.0] ], requires_grad=True)
# x = torch.tensor([ [1.0, 2.0, 3.0],[0.0, 2.0, 2.0], [ 1.0, 4.0, 5.0 ] ], requires_grad = True)
# x = torch.zeros((3,3), requires_grad=True)
y = x
# y = (x + x.T) / 2.0
# y.retain_grad()
sq = symsqrt(y)
print("sq", sq)

sq.sum().backward()
print(f"grad", {x.grad})

# print("scipy root", scipy.linalg.sqrtm( x.detach().numpy() ))