function out = matrix_to_vec(M)
           
    % stacking columns
    k = 1;
    for i=1:size(M,2)    % over columns
        for j=1:size(M,1)% over rows   
            out(k) = M(j,i);
            k = k + 1;
        end
    end
            
end