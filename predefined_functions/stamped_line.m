function path = stamped_line(xi,xf,ti,tf,dt)

    N = ceil(tf-ti)/dt;
    slope = (xf-xi)/N;
    
    time_steps = linspace(0,N,N+1);
    path = xi + time_steps.*slope;

end

