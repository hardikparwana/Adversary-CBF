classdef MatrixVariateDistribution
    
    % x_dot = [f(x) + g(x)]u_bar
    % u_bar = [1; u]; 
    % x: n
    % u: m
    % F(x) = [f(x) g(x)]: n x (1 + m)

    properties(Access = public)
    
        n;
        m;
        M; % n x m
        A; % n x n
        B; % m x m
        
    end
    
    
    methods(Access = public)
        
        function obj = MatrixVariateDistribution(M,A,B);
            
            obj.M = M;
            obj.A = A;
            obj.B = B;
            
            obj.n = size(M,1);
            obj.m = size(M,2);           
            
        end
        
        % probability of given point
        function prob = probability(obj,X)
           
            prob = exp( -0.5*trace( inv(obj.B)*(X-obj.M)' * inv(obj.A) * (X-obj.M) ) )/ ( (2*pi)^(obj.n*obj.m/2) * det(obj.A)^(obj.m/2) * det(obj.B)^(obj.n/2) );
            
        end
        
        % Fit training data
        function out = fit(obj,X,U,X_dot)
            
        end
        
        function out = cov(obj,x,y)           
            
            
        end
        
        % Covariance kernel
        function kernel(obj,x,y)
           
            % return (m + 1) x (m + 1)
            % x: 
            
        end
        
        % predict on test data
        function out = predict(obj,x)
            
            mean = M0 + ( X_dot - M*U )*(  )
            
        end
        
    end
    
end


function


K = kronecker(A,B)