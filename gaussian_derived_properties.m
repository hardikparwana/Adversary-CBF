clear all;
close all;

% Generate random Multivariate Gaussian

for i=1:1:1000
    
    dim_y = 4;

    % Omega Matrix
    Cov = generate_random_cov(dim_y); % 3 vector observations
    Cov_independent = diag( diag(Cov) );
    
    transform_vector = rand(1,dim_y);
    transform_matrix = rand(dim_y,dim_y);
    
    Cov1 = transform_vector * Cov * transform_vector';
    Cov2 = transform_matrix * Cov * transform_matrix';
    
    Cov1_independent = transform_vector * Cov_independent * transform_vector';
    Cov2_independent = transform_matrix * Cov_independent * transform_matrix';
   
    diff1 = Cov1 - Cov1_independent;
    percent_diff = diff1/Cov1*100;
    
    diff2 = trace( Cov2 - Cov2_independent );
    
    if ( any ( (diff1<0)==0 ) || any( (diff1<0)==0 ) )
        fprintf("i = %d \n",i)
        keyboard
    end    
    
end

disp("No exceptions encountered!!")