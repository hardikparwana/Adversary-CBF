import numpy as np

def wrap_angle(angle):
    if angle>np.pi:
        return  angle - 2*np.pi
    elif angle<-np.pi:
        return  angle + 2*np.pi 
    else:
        return angle
    
def euler_to_rot_mat(phi,theta,psi):
    return np.array( [ [ np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(theta)*np.sin(phi),  np.sin(psi)*np.sin(phi)+np.cos(psi)*np.cos(phi)*np.sin(theta) ],
                       [ np.sin(psi)*np.cos(theta),  np.cos(psi)*np.cos(phi)+np.sin(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.sin(phi)+np.sin(theta)*np.sin(psi)*np.cos(phi) ],
                       [ -np.sin(theta),             np.cos(theta)*np.sin(phi)                                    ,  np.cos(theta)*np.cos(phi) ]  ] )
     
def euler_rate_matrix(phi,theta,psi):
    return np.array( [ [ 1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta) ],
                      [ 0,  np.cos(phi)              , -np.sin(phi) ],
                      [ 0,  np.sin(phi)/np.cos(theta), np.cos(phi)*np.sin(theta) ] ] )
    
    
# def compute_trust(A,b,uj,uj_nominal,h,min_dist):
    
#     # distance
#     rho_dist = b - A @ uj;
#     rho_dist = np.tanh(rho_dist); # score between 0 and 1  
    
#     # angle
#     if np.linalg.norm(uj)>0.01:
#         theta_as = np.real(np.arccos( -A @ uj/np.linalg.norm(A)/np.linalg.norm(uj) / 1.05))
#     else:
#         theta_as = np.arccos(0.001)
#     if np.linalg.norm(uj_nominal)>0.01:
#         theta_ns = np.real(np.arccos( -A @ uj_nominal/np.linalg.norm(A)/np.linalg.norm(uj_nominal)/1.05 )) 
#     else:
#         theta_ns = np.arccos(0.001)
#     if (theta_ns<0.05):
#         theta_ns = 0.05

#     rho_theta = np.tanh(theta_ns/theta_as*0.55) # if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
    
#     if rho_dist<0:
#         rho_dist = 0.01
#         print("WARNING: <0")
    
#     # rho_dist and rho_theta both positive
    
#     if (rho_theta>0.5): # can trust to be intact. 
#         if (rho_dist<min_dist):  #Therefore, worst case just slow down
#             trust = 2*rho_theta*rho_dist; # still positive
#         else:
#             trust = 2*rho_theta*rho_dist; # positive
#     else:  # not intact: do not trust
#         if rho_dist<min_dist:  # get away from it as fast as possible. HOWEVER, if h itself is too large, better to not move away too much
#             trust = -2*(1-rho_theta)*(1-rho_dist); # negative
#         else:   # still far away so no need to run away yet but be cautious
#             trust = 2*rho_theta*rho_dist;  # low positive
            
#     return trust

def compute_trust(A,b,uj,uj_nominal,h,min_dist,h_min):
    
    # distance
    rho_dist = b - A @ uj;
    assert(rho_dist>-0.01)
    # assert(h<0.03)
    # assert(-1>0)
    # if h>-h_min:
    #     print(f"small h: {h}")
    rho_dist = np.tanh(rho_dist); # score between 0 and 1  
    
    # angle
    if np.linalg.norm(uj)>0.01:
        theta_as = np.real(np.arccos( -A @ uj/np.linalg.norm(A)/np.linalg.norm(uj) / 1.05))
    else:
        theta_as = np.arccos(0.001)
    if np.linalg.norm(uj_nominal)>0.01:
        theta_ns = np.real(np.arccos( -A @ uj_nominal/np.linalg.norm(A)/np.linalg.norm(uj_nominal)/1.05 )) 
    else:
        theta_ns = np.arccos(0.001)
    if (theta_ns<0.05):
        theta_ns = 0.05

    rho_theta = np.tanh(theta_ns/theta_as*0.55) # if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
    
    if rho_dist<0:
        rho_dist = 0.01
        # print("WARNING: <0")
    
    h_eff = -h
    if h_eff<0:
        h_eff = 0.01
    if h_eff>10:
        h_eff = 10
        
    h_dist = 4*np.tanh(h_eff/np.abs(h_min))
    assert(h_dist>0)
    
    # rho_dist and rho_theta both positive
    # print(f"rho_dist:{rho_dist}, h:{h}, h_min:{h_min} ")
    # if rho_dist>min_dist: # always positive
    #     # trust = 2*rho_theta*(rho_dist-min_dist)/h_eff
    #     trust = 2*rho_theta*(rho_dist-min_dist)/h_dist
    # else: # danger
    #     if h<-h_min:  # far away. therefore still relax/positive
    #         # trust = 2*rho_theta*rho_dist/h_eff
    #         # trust = 2*rho_theta*rho_dist/h_dist
    #         trust = -1*(1-rho_theta)*(rho_dist-min_dist)/h_dist
    #     else:  # definitely negative this time
    #         # print("Negative Trust!")
    #         # trust = -2*(1-rho_theta)*rho_theta
    #         # trust = -2*(1-rho_theta)*rho_dist/h_eff
    #         # trust = -2*(1-rho_theta)*rho_dist/h_dist
    #         trust = 1*(1-rho_theta)*(rho_dist-min_dist)/h_dist
            
    if rho_dist>min_dist: # always positive
        trust = 2*rho_theta*rho_dist#(rho_dist-min_dist)
    else: # danger
        if h<-h_min:  # far away. therefore still relax/positive
            trust = 2*rho_theta*rho_dist
        else:  # definitely negative this time
            # print("Negative Trust!")
            trust = -2*(1-rho_theta)*rho_dist
        
    
    # if (rho_theta>0.5): # can trust to be intact. 
    #     if (rho_dist<min_dist):  #Therefore, worst case just slow down
    #         trust = 2*rho_theta*rho_dist; # still positive
    #     else:
    #         trust = 2*rho_theta*rho_dist; # positive
    # else:  # not intact: do not trust
    #     if rho_dist<min_dist:  # get away from it as fast as possible. HOWEVER, if h itself is too large, better to not move away too much
    #         trust = -2*(1-rho_theta)*(1-rho_dist); # negative
    #     else:   # still far away so no need to run away yet but be cautious
            # trust = 2*rho_theta*rho_dist;  # low positive
            
    return trust