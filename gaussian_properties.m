clear all;
close all;

% Generate random Multivariate Gaussian

% dimension of state
dim = 10;

for i=1:1:1000

    % Generate a random covariance matrix
    while (1)
        a = rand(dim);
        a = (a+a')/2;
        cov_generic = a.*a + 0.5*eye(dim);
        if any( (eig(cov_generic)>0) == 0 )
            continue;   
        else
            break;
        end
    end
%     eig(cov_generic)

    % Generate a digonal covariance matrix
    cov_independent = diag( diag( cov_generic ) );

    means = zeros(dim,1);

    % Mean for conditional distribution
    %x = [y;z];

    dim_y = 4;

    cov_y_generic = cov_generic(1:dim_y,1:dim_y) - cov_generic( 1:dim_y, (dim_y+1):end ) * inv( cov_generic(dim_y + 1:end, dim_y +1:end) ) * cov_generic(dim_y+1:end,1:dim_y);

    cov_y_independent = cov_independent(1:dim_y,1:dim_y) - cov_independent( 1:dim_y, (dim_y+1):end ) * inv( cov_independent(dim_y + 1:end, dim_y +1:end) ) * cov_independent(dim_y+1:end,1:dim_y);

    generic = diag(cov_y_generic);
    independent = diag(cov_y_independent);

    diff = generic - independent;
    
    if any ( (diff<0)==0 )
        keyboard
    end
    
    
end

disp("No exceptions encountered!!")



