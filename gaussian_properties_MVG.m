clear all;
close all;

% Generate random Multivariate Gaussian

for i=1:1:1000
    
    dim_y = 4;

    % Omega Matrix
    Om = generate_random_cov(dim_y); % 2 element vector
    Sigma = generate_random_cov(2); % 3 vector observations
    
    Cov = kron(Sigma,Om);
    
    cov_marginal = Cov(1:dim_y,1:dim_y) - Cov(1:dim_y,dim_y+1:end) * inv( Cov(dim_y+1:end,dim_y+1:end) ) * Cov(dim_y+1:end,1:dim_y);
    
    % Generate a random covariance matrix
    
    % Generate a digonal covariance matrix
%     Cov_independent = kron(Sigma, diag( diag( Cov ) );
    Cov_independent = kron(Sigma, eye(4) );
    cov_marginal_independent = Cov_independent(1:dim_y,1:dim_y) - Cov_independent(1:dim_y,dim_y+1:end) * inv( Cov_independent(dim_y+1:end,dim_y+1:end) ) * Cov_independent(dim_y+1:end,1:dim_y);
    
    generic = diag(cov_marginal);
    independent = diag(cov_marginal_independent);
    
    diff = generic - independent;
    
    if any ( (diff<0)==0 )
        keyboard
    end
    
    
end

disp("No exceptions encountered!!")





