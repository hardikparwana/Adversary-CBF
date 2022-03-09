function cov_generic = generate_random_cov(dim)

    while (1)
        a = 2*rand(dim)-1;
        a = (a+a')/2;
        cov_generic = a.*a + 1.0*eye(dim);
        if any( (eig(cov_generic)>0) == 0 )
            continue;   
        else
            break;
        end
    end
    
        
end