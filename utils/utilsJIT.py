import numpy as np
import torch
    
def wrap_angle_tensor_JIT(angle):
    factor = torch.tensor(2*3.14157,dtype=torch.float)
    if angle>torch.tensor(3.14157):
        angle = angle - factor
    if angle<torch.tensor(-3.14157):
        angle = angle + factor
    return angle

def symsqrt(a):
    """Computes the symmetric square root of a positive definite matrix"""

    s, u = torch.symeig(a, eigenvectors=True)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

    cond = cond_dict[a.dtype]

    above_cutoff = (abs(s) > cond * torch.max(abs(s)))

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]

    B = u @ torch.diag(psigma_diag) @ u.t()

    return B
 
#  traced_sigma_point_expand_JIT = torch.jit.trace( sigma_point_expand_JIT, ( follower_states[i], leader_states[i], leader_weights[i] ) )
#     leader_xdot_states, leader_xdot_weights = sigma_point_expand_JIT( follower_states[i], leader_states[i], leader_weights[i] )       
     
#     traced_sigma_point_scale_up5_JIT = torch.jit.trace( sigma_point_scale_up5_JIT, ( leader_states[i], leader_weights[i] ) )
#     leader_states_expanded, leader_weights_expanded = sigma_point_scale_up5_JIT( leader_states[i], leader_weights[i] )#leader_xdot_weights )
    
#     traced_unicycle_SI2D_UT_Mean_Evaluator = torch.jit.trace(unicycle_SI2D_UT_Mean_Evaluator, (follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded, follower.k_torch, follower.alpha_torch))
#     A, B = traced_unicycle_SI2D_UT_Mean_Evaluator( follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded, follower.k_torch, follower.alpha_torch )
     
        # A, B = traced_unicycle_SI2D_UT_Mean_Evaluator( follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded, follower.k_torch, follower.alpha_torch )
        # A, B = traced_unicycle_SI2D_UT_Mean_Evaluator( follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded, follower.k_torch, follower.alpha_torch )
        # A, B = traced_unicycle_SI2D_UT_Mean_Evaluator( follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded, follower.k_torch, follower.alpha_torch )
        # A, B = traced_unicycle_SI2D_UT_Mean_Evaluator( follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded, follower.k_torch, follower.alpha_torch )
    # traced_get_mean_JIT = torch.jit.trace( get_mean_JIT, (leader_states[i], leader_weights[i] ) )
    # leader_mean_position = traced_get_mean_JIT( leader_states[i], leader_weights[i] )
    
    # traced_unicycle_nominal_input_tensor_jit = torch.jit.trace( unicycle_nominal_input_tensor_jit, ( follower_states[i], leader_mean_position ) )
    # u_ref = traced_unicycle_nominal_input_tensor_jit( follower_states[i], leader_mean_position )
    
    # traced_cbf_controller_layer = torch.jit.trace( cbf_controller_layer, ( u_ref, A, B ) )
    # solution,  = traced_cbf_controller_layer( u_ref, A, B )
    
    
# Time 3: 0.05235576629638672
# Time 3: 0.30892205238342285
# Time 3: 0.001371622085571289
# Time 3: 0.0014166831970214844
# /home/hardik/Desktop/Research/Adversary-CBF/venv/lib/python3.8/site-packages/torch/autograd/__init__.py:173: UserWarning: operator() profile_node %970 : int[] = prim::profile_ivalue(%968)
#  does not have profile information (Triggered internally at  ../torch/csrc/jit/codegen/cuda/graph_fuser.cpp:108.)
#   Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# grads: alpha:[[4.4957764e-02 4.0598109e-02 1.4281761e-08]], k:9.309481356467586e-06
# reward computation: f:[[-0.29391723 -0.00201392  0.17721192]], L:[[0.55       0.30582627]]
# Time 3: 0.043560028076171875
# Time 3: 0.19827055931091309
# Time 3: 0.14853596687316895
# Time 3: 0.001374959945678711
