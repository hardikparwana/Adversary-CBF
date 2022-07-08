import numpy as np
import torch
from utils.utils import wrap_angle

class Unicycle:
    
    def __init__(self,X0,dt,ax,id,num_robots=1,num_adversaries = 1, alpha=0.8,color='r',palpha=1.0,plot=True, identity='nominal'):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Unicycle'
        self.identity = identity # or adversary
        
        self.X = X0.reshape(-1,1)
        self.X_nominal = np.copy(self.X)
        self.dt = dt
        self.id = id
        
        self.U = np.array([0,0]).reshape(-1,1)
        self.x_dot_nominal = np.array([ [0],[0],[0] ])
        self.U_ref = np.array([0,0]).reshape(-1,1)
        self.U_ref_nominal = np.copy(self.U)
        
        # Plot handles
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],alpha=palpha,s=40,facecolors='none',edgecolors=color) #,c=color
            self.radii = 0.5
            self.palpha = palpha
            if palpha==1:
                self.axis = ax.plot([self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
            self.render_plot()
            
            self.body_nominal = ax.scatter([],[],alpha=0.1,s=40,facecolors='none',edgecolors=color) #,c=color
            self.render_plot(mode='nominal')
            
            
        
        
        self.alpha = alpha*np.ones((num_robots,1))
        # for Trust computation
        self.adv_alpha =  alpha*np.ones((1,num_adversaries))# alpha*np.ones((1,num_adversaries))
        self.trust_adv = np.ones((1,num_adversaries))
        self.robot_alpha = alpha*np.ones((1,num_robots))
        self.trust_robot = np.ones((1,num_robots))
        self.adv_objective = [0] * num_adversaries
        self.robot_objective = [0] * num_robots
        self.robot_h = np.ones((1,num_robots))
        self.adv_h = np.ones((1,num_adversaries))
        
        # Old
        # self.adv_alpha =  alpha*np.ones(num_adversaries)# alpha*np.ones((1,num_adversaries))
        # self.trust_adv = 1
        # self.robot_alpha = alpha*np.ones(num_robots)
        # self.trust_robot = 1
        # self.adv_objective = [0] * num_adversaries
        # self.robot_objective = [0] * num_robots
        
        num_constraints1  = num_robots - 1 + num_adversaries
        self.A1 = np.zeros((num_constraints1,2))
        self.b1 = np.zeros((num_constraints1,1))
        self.A2 = np.zeros((num_constraints1,2))
        self.b2 = np.zeros((num_constraints1,1))
        
        # For plotting
        self.adv_alphas = alpha*np.ones((1,num_adversaries))
        self.trust_advs = np.ones((1,num_adversaries))
        self.robot_alphas = alpha*np.ones((1,num_robots))
        self.trust_robots = 1*np.ones((1,num_robots))
        self.Xs = X0.reshape(-1,1)
        self.Us = np.array([0,0]).reshape(-1,1)
        self.adv_hs = np.ones((1,num_adversaries))
        self.robot_hs = np.ones((1,num_robots))
        
        ## Store state
        self.X_org = np.copy(self.X)
        self.U_org = np.copy(self.U)
        
        self.Xs = np.copy(self.X)
        self.Xdots = np.array([0,0,0]).reshape(-1,1)
        gp = []
        
    def f_torch(self,x):
        return torch.tensor(np.array([0.0,0.0,0.0]).reshape(-1,1),dtype=torch.float)

    def g_torch(self,x):
        # return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])
        g1 = torch.cat( ( torch.cos(x[2,0]).reshape(-1,1),torch.tensor([[0]]) ), axis=1 )
        g2 = torch.cat( ( torch.tensor([[0]]), torch.sin(x[2,0]).reshape(-1,1) ), axis=1 )
        g3 = torch.tensor([[0,1]],dtype=torch.float)
        gx = torch.cat((g1,g2,g3))
        return gx
    
    def f(self, mode='actual'):
        if mode=='actual':
            return np.array([0,0,0]).reshape(-1,1)
        if mode=='nominal':
            return np.array([0,0,0]).reshape(-1,1)
    
    def g(self, mode='actual'):
        if mode=='actual':
            return np.array([ [ np.cos(self.X[2,0]), 0 ],
                          [ np.sin(self.X[2,0]), 0],
                          [0, 1] ])    
        if mode=='nominal':
            return np.array([ [ np.cos(self.X_nominal[2,0]), 0 ],
                          [ np.sin(self.X_nominal[2,0]), 0],
                          [0, 1] ]) 
            
         
    def step(self,U, dt, mode='actual'): 
        if mode=='actual':
            self.U = U.reshape(-1,1)
            self.X = self.X + ( self.f() + self.g() @ self.U )*dt
            self.X[2,0] = wrap_angle(self.X[2,0])
            # self.Xs = np.append(self.Xs,self.X,axis=1)
            # self.Us = np.append(self.Us,self.U,axis=1)
            return self.X
        if mode=='nominal':
            self.U_nominal = U.reshape(-1,1)
            self.X_nominal = self.X_nominal + ( self.f(mode='nominal') + self.g(mode='nominal') @ self.U_nominal )*dt
            self.X_nominal[2,0] = wrap_angle(self.X_nominal[2,0])
            # self.Xs = np.append(self.Xs,self.X,axis=1)
            # self.Us = np.append(self.Us,self.U,axis=1)
            return self.X_nominal
    
    def step_torch(self, X, U, dt):
        return X + ( self.f_torch(X) + self.g_torch(X) @ U ) * dt
    
    def render_plot(self, mode='actual'):
        
        if self.plot:
            if mode=='actual':
                x = np.array([self.X[0,0],self.X[1,0]])
                self.body.set_offsets([x[0],x[1]])
                if self.palpha==1:
                    self.axis[0].set_ydata([self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
                    self.axis[0].set_xdata( [self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])] )
            elif mode=='nominal':
                x = np.array([self.X_nominal[0,0],self.X_nominal[1,0]])
                self.body_nominal.set_offsets([x[0],x[1]])
        # self.axis = ax.plot([self.X[0,0],self.X[0,0]+np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+np.sin(self.X[2,0])])
        
    def lyapunov(self, G):
        V = np.linalg.norm( self.X[0:2] - G[0:2] )**2
        dV_dx = np.append( 2*( self.X[0:2] - G[0:2] ).T, [[0]], axis=1)
        return V, dV_dx
    
    def lyapunov_tensor(self, X, G):
        V = torch.square ( torch.norm( X[0:2] - G[0:2] ) )
        dV_dx = torch.cat ( (2*( X[0:2] - G[0:2] ).T, [[0]] ), 1) 
        return V, dV_dx
    
    def nominal_input(self,G):
        # V, dV_dx = self.lyapunov(G)
        #Define gamma for the Lyapunov function
        k_omega = 2.0 #0.5#2.5
        k_v = 2.0 #0.5
        theta_d = np.arctan2(G.X[1,0]-self.X[1,0],G.X[0,0]-self.X[0,0])
        error_theta = wrap_angle( theta_d - self.X[2,0] )

        omega = k_omega*error_theta

        distance = max(np.linalg.norm( self.X[0:2,0]-G.X[0:2,0] ) - 0.3,0)

        v = k_v*( distance )*np.cos( error_theta )
        return np.array([v, omega]).reshape(-1,1) #np.array([v,omega])
    
    def nominal_input_tensor(self, X, targetX):
        #simple controller for now: considers estimated disturbance
        # print(X,target.X)
        #Define gamma for the Lyapunov function
        k_omega = 2.0 #0.5#2.5
        k_v = 2.0 #0.5
        diff = targetX[0:2,0] - X[0:2,0]

        theta_d = torch.atan2(targetX[1,0]-X[1,0],targetX[0,0]-X[0,0])
        error_theta = self.wrap_angle_tensor( theta_d - X[2,0] )

        omega = k_omega*error_theta 
        v = k_v*( torch.norm(diff) ) * torch.cos( error_theta )
        v = v.reshape(-1,1)
        omega = omega.reshape(-1,1)
        U = torch.cat((v,omega))
        return U
    
    # def agent_barrier(self,agent,d_min):
    #     h = d_min**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
    #     dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1)
        
    #     if agent.type=='SingleIntegrator2D':
    #         dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
    #     elif agent.type=='Unicycle':
    #         dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1 )
    #     elif agent.type=='FixedWing':
    #         print("TO DO here!!!!")
        
    #     return h, dh_dxi, dh_dxj
    
    def sigma(self,s):
        k1 = 2.0
        return (np.exp(k1-s)-1)/(np.exp(k1-s)+1)
    
    def sigma_torch(self,s):
        k1 = 2.0
        return torch.div( (torch.exp(k1-s)-1) , (torch.exp(k1-s)+1) )
    
    def sigma_der(self,s):
        k1 = 2
        return -np.exp(k1-s)/( 1+np.exp( k1-s ) ) * ( 1 + self.sigma(s) )
    
    def sigma_der_torch(self,s):
        k1 = 2
        return -torch.div ( torch.exp(k1-s),( 1+torch.exp( k1-s ) ) ) @ ( 1 + self.sigma_torch(s) )
    
    def agent_barrier(self,agent,d_min):
        beta = 1.01
        h = beta*d_min**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
        h1 = h
        
        theta = self.X[2,0]
        s = (self.X[0:2] - agent.X[0:2]).T @ np.array( [np.sin(theta),np.cos(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        # print(f"h1:{h1}, h2:{h}")
        # assert(h1<0)
        der_sigma = self.sigma_der(s)
        dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T - der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ),  - der_sigma * ( np.cos(theta)*( self.X[0,0]-agent.X[0,0] ) - np.sin(theta)*( self.X[1,0] - agent.X[1,0] ) ) , axis=1)
        
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T # row
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), np.array([[0]]), axis=1 )
        else:
            dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
        
        return h, dh_dxi, dh_dxj
    
    def agent_barrier_torch(self, X, targetX, d_min, target_type='Unicycle'):
        beta = 1.01
        h = beta*d_min**2 -  torch.square( torch.norm(X[0:2] - targetX[0:2])  )
        h1 = h
        
        theta = X[2,0]
        s = (X[0:2] - targetX[0:2]).T @ torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
        h_final = h - self.sigma_torch(s)
        # print(f"h1:{h1}, h2:{h}")
        # assert(h1<0)
        der_sigma = self.sigma_der_torch(s)
        dh_dxi =     torch.cat( ( -2*( X[0:2] - targetX[0:2] ).T - der_sigma @ torch.cat( (torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1)),1 ) ,  - der_sigma * ( torch.cos(X[2,0]).reshape(-1,1) @ ( X[0,0]-targetX[0,0] ).reshape(-1,1) - torch.sin(X[2,0]).reshape(-1,1) @ ( X[1,0] - targetX[1,0] ).reshape(-1,1) ) ), 1)
        
        if target_type=='SingleIntegrator2D':
            dh_dxj = 2*( X[0:2] - targetX[0:2] ).T # row
        elif target_type=='Unicycle':
            dh_dxj = torch.cat( ( -2*( X[0:2] - targetX[0:2] ).T + der_sigma @ torch.cat( (torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1)),1 ) , torch.tensor([[0]]) ) , 1)
        else:
            dh_dxj = 2*( X[0:2] - targetX[0:2] ).T
        
        return h_final, dh_dxi, dh_dxj
    
    def compute_reward(self,X,targetX):
        return torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) )
    