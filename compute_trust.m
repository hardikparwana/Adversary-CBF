%% V1
% function trust = compute_trust(A,b,uj,uj_nominal,min_dist)
%             
%        rho_dist = b - A*uj - min_dist;
%        if rho_dist<0
%            keyboard
%        end
%        rho_dist = -1 + 2*tanh(rho_dist);
%        theta_as = real(acos( -A*uj/norm(A)/norm(uj) ));
%        theta_ns = real(acos( -A*uj_nominal/norm(A)/norm(uj_nominal) )); 
%        if (theta_ns<0.05)
%            theta_ns = 0.05;
%        end
% 
%        rho_theta = -1 + 2*tanh(theta_ns/theta_as);    
% 
%        trust = rho_dist * rho_theta;
%        
%        if ((rho_dist<0) && (rho_theta<0))
%                trust = -trust;
%        end
%            
% end

%% V2
% function trust = compute_trust(A,b,uj,uj_nominal,min_dist)
%           
%        % distance
%        rho_dist = b - A*uj - min_dist;
% %        if rho_dist<-min_dist
% %            keyboard
% %        end
%        rho_dist = tanh(rho_dist); % score between 0 and 1  
%        
%        % angle
%        theta_as = real(acos( -A*uj/norm(A)/norm(uj) ));
%        theta_ns = real(acos( -A*uj_nominal/norm(A)/norm(uj_nominal) )); 
%        if (theta_ns<0.05)
%            theta_ns = 0.05;
%        end
%        rho_theta = tanh(theta_ns/theta_as); % if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
% 
%        trust = -1 + 2* rho_dist * rho_theta;
%        
%        if ((rho_dist<0) && (rho_theta<0))
%                trust = -trust;
%        end
%            
% end

%% V3 from video: sensitive to max rate of change of alpha: issue: rho_dist is negative
% function trust = compute_trust(A,b,uj,uj_nominal,min_dist)
%           
%        % distance
%        rho_dist = b - A*uj - min_dist;
% %        if rho_dist<0
% %            keyboard
% %        end
%        rho_dist = tanh(rho_dist); % score between 0 and 1  
%        
%        % angle
%        theta_as = real(acos( -A*uj/norm(A)/norm(uj) ));
%        theta_ns = real(acos( -A*uj_nominal/norm(A)/norm(uj_nominal) )); 
%        if (theta_ns<0.05)
%            theta_ns = 0.05;
%        end
%        rho_theta = -1 + 2*tanh(theta_ns/theta_as); % if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
%        
%        trust = rho_dist * rho_theta;
%                   
% end

%% V4: based on distance to current alpha and alpha=0
% works same as V3
% try this with 1.5 human velocity
% function trust = compute_trust(A,b,uj,uj_nominal,min_dist)
%           
%        % distance
%        rho_dist = b - A*uj - min_dist;
% %        if rho_dist<0
% %            keyboard
% %        end
%        rho_dist = tanh(rho_dist); % score between 0 and 1  
%        
%        % angle
%        theta_as = real(acos( -A*uj/norm(A)/norm(uj) ));
%        theta_ns = real(acos( -A*uj_nominal/norm(A)/norm(uj_nominal) )); 
%        if (theta_ns<0.05)
%            theta_ns = 0.05;
%        end
%        rho_theta = -1 + 2*tanh(theta_ns/theta_as) % if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
%        
%        if rho_dist<0
%            if rho_theta>0 % then relax, but ideally check if worth relaxing
%                % if (will not collide befor reaching target)
%                trust = -rho_theta*rho_dist;
%            else
%                % again all adversarial intentions but if 
%                trust = -rho_theta*rho_dist;
%            end
%        else
%            if rho_theta>0
%                trust = rho_theta*rho_dist;
%            else
%                trust = -rho_theta*rho_dist;
%            end
%        end
% 
% %        trust = rho_dist * rho_theta;
%                   
% end

%% V5: based on threshold and angle: incorrect: based on thresholds
function trust = compute_trust(A,b,uj,uj_nominal,h,min_dist)
          
       % distance
       rho_dist = b - A*uj;
       rho_dist = tanh(rho_dist); % score between 0 and 1  
       
       % angle
       theta_as = real(acos( -A*uj/norm(A)/norm(uj) ));
       theta_ns = real(acos( -A*uj_nominal/norm(A)/norm(uj_nominal) )); 
       if (theta_ns<0.05)
           theta_ns = 0.05;
       end
       rho_theta = tanh(theta_ns/theta_as*0.55); % if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
       
       if rho_dist<0
           rho_dist = 0.01;
           disp("WARNING: <0")
       end
       
       % rho_dist and rho_theta both positive
       
       if (rho_theta>0.5) % can trust to be intact. 
           if (rho_dist<min_dist)  %Therefore, worst case just slow down
               trust = 2*rho_theta*rho_dist; % still positive
           else
               trust = 2*rho_theta*rho_dist; % positive
           end
       else  % not intact: do not trust
           if rho_dist<min_dist  % get away from it as fast as possible. HOWEVER, if h itself is too large, better to not move away too much
               trust = -2*(1-rho_theta)*(1-rho_dist); % negative
           else   % still far away so no need to run away yet but be cautious
               trust = 2*rho_theta*rho_dist;  % low positive
           end
       end
                  
end

%% V5: based angle/distance
% function trust = compute_trust(A,b,uj,uj_nominal,min_dist)
%           
%        % distance
%        rho_dist = b - A*uj;
%        rho_dist = tanh(rho_dist); % score between 0 and 1  
%        
%        % angle
%        theta_as = real(acos( -A*uj/norm(A)/norm(uj) ));
%        theta_ns = real(acos( -A*uj_nominal/norm(A)/norm(uj_nominal) )); 
%        if (theta_ns<0.05)
%            theta_ns = 0.05;
%        end
%        rho_theta = -1 + 2*tanh(theta_ns/theta_as); % if it is close to it's nominal, then trust high. if far away from it's nominal, then trust low     
%        
%        if rho_dist<0
%            rho_dist = 0.01;
%            disp("WARNING: <0")
%        end
%        
%        if (rho_theta>0)
%            if (rho_dist<min_dist) % urgent
%                trust = -rho_theta*rho_dist;
%            else
%                trust = rho_theta*rho_dist;
%            end
%        else
%            if rho_dist<min_dist  % urgent
%                trust = rho_theta*rho_dist;
%            else
%                trust = -rho_theta*rho_dist;
%            end
%        end
%                   
% end