import torch
import numpy as np
# from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator1D import *
# from robot_models.DoubleIntegrator2D import *

class torch_dynamics(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gp_model, input_data):

        # x_{t+1} = f(x_t) + g(x_t)u_t
        x = input_x
        u = input_u
        fx = fx_(x)  
        gx = gx_(x)  
        
        # for gradient computation
        df_dx =  df_dx_(x) 
        dgxu_dx = dgxu_dx_(x, u)  
            
        # save tensors for use in backward computation later on
        ctx.save_for_backward(x,u, gx, df_dx, dgxu_dx)
        # print(f"fx:{fx}, gx:{gx}, u:{u}")
        return fx + gx * u

    @staticmethod
    def backward(ctx, grad_output):
        '''
        grad_output is column vector. Math requires it to be row vector so need adjustment in returning the values
        '''

        input_x, input_u, gx, df_dx, dgxu_dx, = ctx.saved_tensors
        x = input_x.detach().numpy()
        u = input_u.detach().numpy()

        n_x = np.shape(x)
        n_u = np.shape(u)
               
        # print(grad_output)
        
        gradient_x = df_dx + dgxu_dx
        gradient_u =  gx 

        output_grad_x = grad_output *gradient_x
        output_grad_u = grad_output *gradient_u

        return output_grad_x, output_grad_u