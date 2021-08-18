classdef A <handle
  properties
      eg_var
  end
    methods
        function multi_egvar(obj,n)
              obj.eg_var = obj.eg_var*n;
        end
        function multi_n_3(obj,n)
            for ik = 1:3
                obj.multi_egvar(n);
            end
        end
    end
 end