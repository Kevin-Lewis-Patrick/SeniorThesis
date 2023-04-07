using PyPlot, LinearAlgebra, Distributed, Arpack, ProgressMeter, JLD, SparseArrays, SharedArrays
@everywhere import Printf.@printf
@everywhere f(x,o) = 0
@everywhere function b(x,o)
    if (x[2] < 0)
        return 1
    else
        return 1
    end
end

@everywhere function Coarse()
    nx = 200;
    ny = 200;
    nw = 8;
    mu_a = 3;
    mu_s = 0
    NN = 20;

    X = range(-1,stop=1,length=nx);
    Y = range(-1,stop=1,length=ny);
    W = range(0,stop=2*pi-(2*pi)/nw,length=nw)

    """
    U_Coarse lays out the following coarse sub-grid (% is an approximated point):

    % * % * %
    * * * * *
    % * % * %
    * * * * *
    % * % * %

    All asterisks (*) are interpolated afterwards.

    """

    U_Coarse = TransportEquation(X[1:2:end],Y[1:2:end],W[1:2:end],mu_a,mu_s,f,b,NN);
    U = zeros(nx,ny,nw);
    for i=1:2:nx
        for j=1:2:ny
            for k=1:2:nw
                U[i,j,k] = U_Coarse[Int64((i+1)/2),Int64((j+1)/2),Int64((k+1)/2)];
            end
        end
    end
    interp(x) = interpolant_intermediate(U_Coarse,size(U_Coarse)[1],size(U_Coarse)[2],size(U_Coarse)[3],x);
    for i=2:nx
        for j=2:ny
            for k=2:nw
                if (((i % 2 == 0) || (j % 2 == 0) || (k % 2 == 0)) && in_Domain(X[i],Y[j]))
                    U[i,j,k] = interp([ X[i] , Y[j] , W[k] ]);
                end
            end
        end
    end
    return U
end

# nx, ny - spatial resolutions
# nw = directional / "temporal" resolution (which directions do light scatter in), generates even # of points from 0 to 2*pi inclusive
# mu_a - absorption coefficient
# mu_s - scattering coefficient
# forcing - forcing function
# boundary - boundary conditions
# nprocs - number of active processors (parallelization)
@everywhere function TransportEquation_Scatter(nx,ny,nw,mu_a,mu_s,forcing,boundary,nprocs=nprocs())
    X = range(-1,stop=1,length=nx);
    Y = range(-1,stop=1,length=ny);
    W = range(0,stop=2*pi-(2*pi)/nw,length=nw);
    
    # If no scattering, run TransportEquation_P
    if (mu_s == 0)
        return TransportEquation_P(nx,ny,nw,mu_a,mu_s,forcing,boundary,nprocs)
    end

    U_Prev = zeros(size(X)[1],size(Y)[1],size(W)[1])
    i=0
    ninf=1000
    tol=0.01
    U_Prev = zeros(nx,ny,nw)


    for i=1:nx
        for j=1:ny
            for w=1:nw
                U_Prev[i,j,w] = forcing([X[i],Y[j]],W[w]) + boundary([X[i],Y[j]],W[w])
            end
        end
    end


    while (ninf > tol)
        if (nprocs==1)
            interp(x) = interpolant_intermediate(U_Prev,nx,ny,nw,x)
            x = @spawn fill_SU(X,Y,W,U_Prev,mu_s,interp)
            Su = fetch(x)
        else
            Su = fill_SU_P(nx,ny,nw,U_Prev,mu_s,nprocs)
        end
        S(x) = interpolant_intermediate(Su,nx,ny,nw,x)
        function new_forcing(x,omega)
            return forcing(x,omega) + S([x[1],x[2],omega])
        end
        U=zeros(nx,ny,nw)
        if (nprocs == 1)
            Ux = @spawn TransportEquation(X,Y,W,mu_a,mu_s,new_forcing,boundary,20)
            U = fetch(Ux)
        else
            U = TransportEquation_P(nx,ny,nw,mu_a,mu_s,new_forcing,boundary,nprocs)
        end
        ninf = norm(U-U_Prev,Inf)
        println("Inf norm: ", ninf)
        U_Prev = Array(U)
    end
    return U_Prev
end

# "Master Function" when using multiple cores or processes. 
@everywhere function TransportEquation_P(nx,ny,nw,mu_a,mu_s,f,b,nprocs=nprocs())
    #println("nx is: ", nx)
    X = range(-1,stop=1,length=nx);
    Y = range(-1,stop=1,length=ny);
    W = range(0,stop=2*pi-(2*pi)/nw,length=nw)

    U = zeros(nx,ny,nw)#TransportEquation_P(X,Y,W,0.3,0.1,f,b,20);

    N = Int64(nw)
    remainder = rem(N,nprocs-1)
    array_of_sizes = Array{Int64}(zeros(nprocs-1))

    for i=1:size(array_of_sizes)[1]
        array_of_sizes[i] = Int64(div(N,nprocs-1))
        if (i <= remainder)
            array_of_sizes[i] = Int64(array_of_sizes[i] + 1)
        end
    end
    r = [];
    begin
        for i=1:nprocs-2
            starting = 1 + (i - 1) * array_of_sizes[i]
            ending = starting + array_of_sizes[i] - 1
            x = @spawn TransportEquation(X,Y,W[starting:ending],mu_a,mu_s,f,b,20)
            push!(r,x);
        end
        starting = N - array_of_sizes[nprocs - 1] + 1
        x = @spawn TransportEquation(X,Y,W[starting:end],mu_a,mu_s,f,b,20)
        push!(r,x)
        w_counter = 1

        for i = 1:nprocs-2
            x = fetch(r[i])
            for j = 1:array_of_sizes[i]
                U[:,:,w_counter] = x[:,:,j]
                w_counter = w_counter + 1
            end
        end
        x = fetch(r[nprocs-1])
        for j = 1:array_of_sizes[nprocs-1]
            U[:,:,w_counter] = x[:,:,j]
            w_counter = w_counter + 1
        end
    end
    return U
end

# Fill matrix used when implementing scattering. Parallelized function.
@everywhere function fill_SU_P(nx,ny,nw,U_Prev,mu_s,nprocs)
    X = range(-1,stop=1,length=nx); Y = range(-1,stop=1,length=ny); W = range(0,stop=2*pi-(2*pi)/nw,length=nw);
    interpolation(x) = interpolant_intermediate(U_Prev,nx,ny,nw,x)
    Su = zeros(nx,ny,nw)
    N = Int64(nw)
    remainder = rem(N,nprocs-1)
    array_of_sizes = Array{Int64}(zeros(nprocs-1))

    for i=1:size(array_of_sizes)[1]
        array_of_sizes[i] = Int64(div(N,nprocs-1))
        if (i <= remainder)
            array_of_sizes[i] = Int64(array_of_sizes[i] + 1)
        end
    end
    s = [];
    begin
        for i=1:nprocs-2
            starting = 1 + (i - 1) * array_of_sizes[i]  #1,3,5,7
            ending = starting + array_of_sizes[i] - 1 #2,4,6,8
            x = @spawn fill_SU(X,Y,W[starting:ending],U_Prev,mu_s,interpolation)
            push!(s,x);
        end
        starting = N - array_of_sizes[nprocs - 1] + 1
        x = @spawn fill_SU(X,Y,W[starting:end],U_Prev,mu_s,interpolation)
        push!(s,x);
        w_counter = 1
        #println("size of r: ", size(r))
        for i = 1:nprocs-2
            x = fetch(s[i])
            for j = 1:array_of_sizes[i]
                Su[:,:,w_counter] = x[:,:,j]
                w_counter = w_counter + 1
            end
        end
        x = fetch(s[nprocs-1])
        for j = 1:array_of_sizes[nprocs-1]
            Su[:,:,w_counter] = x[:,:,j]
            w_counter = w_counter + 1
        end
    end
    return Su
end

# Fill matrix used when enabling scattering term
@everywhere function fill_SU(X,Y,W,U_Prev,mu_s,interp)
    Su = zeros(size(X)[1],size(Y)[1],size(W)[1]);
    for i=1:size(X)[1]
        for j=1:size(Y)[1]
            for k=1:size(W)[1]
                inside_Integral(wp) = interp([X[i],Y[j],wp]) * scattering(W[k],wp)
                Su[i,j,k] = mu_s * quadrature(inside_Integral,0,2*pi,20)
            end
        end
    end
    return Su
end

# Different method for calculating Transport Equation, using sparse matrices for greater time efficiency but lessened space efficiency.
@everywhere function TransportEquationMatrix(nx,ny,nw,mu_a,mu_s,forcing,boundary)

    X = range(-1.01,stop = 1.01,              length = nx);
    Y = range(-1.01,stop = 1.01,              length = ny);
    W = range(0, stop = 2*pi-(2*pi/nw), length = nw);

    delx = X[2]-X[1]
    dely = Y[2]-Y[1]
    triplet_Size = size(X)[1] * size(Y)[1] * size(W)[1]
    U =      zeros(size(X)[1],size(Y)[1],size(W)[1])
    A =      spzeros(triplet_Size, triplet_Size)
    f = zeros((triplet_Size,1))

    Classification=zeros(size(U));

    # max possible allocations
    row = zeros((5+nw) * (nx * ny * nw))
    col = zeros((5+nw) * (nx * ny * nw))
    val = zeros((5+nw) * (nx * ny * nw))
    N = 1;
    for i=1:triplet_Size
        point = rli(i,size(U))
        x=floor(Int64,point[1]); y=floor(Int64,point[2]); w=floor(Int64,point[3])

        if (!in_Domain(X[x],Y[y])) && (in_Domain(X[x] + delx,Y[y]) || in_Domain(X[x] - delx,Y[y]) || in_Domain(X[x],Y[y] + dely) || in_Domain(X[x],Y[y]-dely))
            # Inflow boundary w.r.t direction omega
            if (cos(W[w])*X[x]+sin(W[w])*Y[y] < 0)
                Classification[y,x,w]=1;
                f[i] = boundary([X[x],Y[y]],W[w])

                #A[i,i] = 1
                row[N] = i;
                col[N] = i;
                val[N] = 1;
                N = N+1

            else #Outflow Boundary

                omega = W[w]

                omega = Base.atan(Y[y],X[x])
                if (omega < 0)
                    omega = omega + 2*pi
                end

                if (abs(omega) < pi/200) #close to 0

                    row[N] = i;
                    col[N] = i;
                    val[N] = cos(omega)/delx+(mu_a+mu_s);

                    N=N+1

                    row[N] = i;
                    col[N] = li(x-1,y,w,size(U));
                    val[N] = -cos(omega)/delx;

                    N = N+1

                elseif (abs(omega - pi/2) < pi/200) #close to pi/2

                    row[N] = i;
                    col[N] = i;
                    val[N] = sin(omega)/dely+(mu_a+mu_s);

                    N=N+1

                    row[N] = i;
                    col[N] = li(x,y-1,w,size(U));
                    val[N] = -sin(omega)/dely;

                    N=N+1

                elseif (abs(omega - pi) < pi/200) #close to pi

                    row[N] = i;
                    col[N] = i;
                    val[N] = -cos(omega)/delx+(mu_a+mu_s);

                    N=N+1

                    row[N] = i;
                    col[N] = li(x+1,y,w,size(U));
                    val[N] = cos(omega)/delx;

                    N=N+1

                elseif (abs(omega - 3*pi/2) < pi/200) #close to 3*pi/2

                    row[N] = i;
                    col[N] = i;
                    val[N] = -sin(omega)/dely + (mu_a+mu_s);

                    N=N+1

                    row[N] = i;
                    col[N] = li(x,y+1,w,size(U));
                    val[N] = sin(omega)/dely;

                    N=N+1

                elseif (omega > 0) && (omega < pi/2) # 1

                    row[N] = i;
                    col[N] = i;
                    val[N] = cos(omega)/delx + sin(omega)/dely + (mu_a + mu_s);

                    N=N+1

                    row[N] = i;
                    col[N] = li(x-1,y,w,size(U));
                    val[N] = -cos(omega)/delx;

                    N=N+1

                    row[N] = i;
                    col[N] = li(x,y-1,w,size(U));
                    val[N] = -sin(omega)/dely;

                    N=N+1

                elseif (omega > pi/2) && (omega < pi) # 2

                    row[N] = i;
                    col[N] = i;
                    val[N] = -cos(omega)/delx + sin(omega)/dely + (mu_a + mu_s);

                    N=N+1

                    row[N] = i;
                    col[N] = li(x+1,y,w,size(U));
                    val[N] = cos(omega)/delx;

                    N=N+1

                    row[N] = i;
                    col[N] = li(x,y-1,w,size(U));
                    val[N] = -sin(omega)/dely;

                    N=N+1

                elseif (omega > pi) && (omega < 3*pi/2)# 3

                    row[N] = i;
                    col[N] = i;
                    val[N] = -cos(omega)/delx - sin(omega)/dely + (mu_a + mu_s);

                    N=N+1

                    row[N] = i;
                    col[N] = li(x+1,y,w,size(U));
                    val[N] = cos(omega)/delx;

                    N=N+1

                    row[N] = i;
                    col[N] = li(x,y+1,w,size(U));
                    val[N] = sin(omega)/dely;

                    N=N+1

                else # 4

                    row[N] = i;
                    col[N] = i;
                    val[N] = cos(omega)/delx - sin(omega)/dely + (mu_a + mu_s);

                    N=N+1

                    row[N] = i;
                    col[N] = li(x-1,y,w,size(U));
                    val[N] = -cos(omega)/delx;

                    N=N+1

                    row[N] = i;
                    col[N] = li(x,y+1,w,size(U));
                    val[N] = sin(omega)/dely;

                    N=N+1

                end

                for j=1:size(U)[3]
                    #A[i,li(x,y,j,size(U))] -= 2*pi*mu_s * scattering(W[w],W[j])/size(W)[1]
                    row[N] = i;
                    col[N] = li(x,y,j,size(U));
                    val[N] = -2*pi*mu_s * scattering(W[w],W[j])/size(W)[1];

                    N=N+1
                end

                f[i] = forcing([X[x],Y[y]],omega)

            end
        elseif ((X[x]^2 + Y[y]^2) > 1) #Outside of boundary
            #A[i,i] = 1
            f[i] = 0
            row[N] = i;
            col[N] = i;
            val[N] = 1;

            N=N+1

        else #Inside Boundary. Try all central difference at first

            row[N] = i;
            col[N] = li(x,y+1,w,size(U));
            val[N] = sin(W[w]) / (2*dely);

            N=N+1

            row[N] = i;
            col[N] = li(x,y-1,w,size(U));
            val[N] = -sin(W[w]) / (2*dely);

            N=N+1

            row[N] =  i;
            col[N] = li(x+1,y,w,size(U));
            val[N] = cos(W[w]) / (2*delx);

            N=N+1

            row[N] = i;
            col[N] = li(x-1,y,w,size(U));
            val[N] = -cos(W[w]) / (2*delx);

            N=N+1

            for j=1:size(U)[3]
                #A[i,li(x,y,j,size(U))] -= 2*pi*mu_s * scattering(W[w],W[j])/size(W)[1]
                row[N] = i;
                col[N] = li(x,y,j,size(U));
                val[N] = -2*pi*mu_s * scattering(W[w],W[j])/size(W)[1];
                N=N+1
            end

            #A[i,i] += (mu_s + mu_a)
            row[N] = i;
            col[N] = i;
            val[N] = mu_s + mu_a;
            N=N+1
            f[i] = forcing([X[x],Y[y]],W[w])
        end

    end

    row = row[1:N-1]
    col = col[1:N-1]
    val = val[1:N-1]
    A = SparseArrays.sparse(row,col,val)
    u_Sol = A \ f

    for i=1:size(u_Sol)[1]
        triplet = rli(i,size(U))
        x = floor(Int64,triplet[1])
        y = floor(Int64,triplet[2])
        z = floor(Int64,triplet[3])
        if (X[x]^2 + Y[y]^2 <= 1)
            U[x,y,z] = u_Sol[i]
        end
    end

    return U
end
## Do not call, this is called by TransportEquation_P for parallelizing
@everywhere function TransportEquation(X,Y,W,mu_a,mu_s,forcing,boundary,NN)
    mu_t = mu_a + mu_s

    sizeX = size(X)[1]
    sizeY = size(Y)[1]
    sizeW = size(W)[1]
    U = zeros(Float64, sizeX, sizeY, sizeW)
    for i=1:sizeX
        for j=1:sizeY
            actualX = X[i]
            actualY = Y[j]
            if (in_Domain(actualX,actualY))
            for w=1:sizeW
                    omega = W[w]

                    w1 = cos(omega)
                    w2 = sin(omega)

                    a = w1^2 + w2^2
                    b = 2*actualX*w1 + 2*actualY*w2
                    c = actualX^2 + actualY^2-1

                    t1 = (-b + sqrt(b^2 - 4*a*c))/(2*a)
                    t2 = (-b - sqrt(b^2 - 4*a*c))/(2*a)
                    t0 = min(t1,t2)

                    if (t0 > 0)
                        println("You messed up somewhere, these shouldn't be negative.")
                        t0 = max(t1,t2)
                    end

                    x0 = actualX + t0 * w1
                    y0 = actualY + t0 * w2

                    distance = sqrt((x0 - actualX)^2 + (y0 - actualY)^2)

                    gamma(s) = [x0, y0] + s*[w1,w2]
                    inside_integral(s) = exp(mu_t*s) * forcing(gamma(s),omega) # 50

                    U[i,j,w] += exp(-mu_t*distance) * (quadrature(inside_integral,0,distance,NN) + boundary([x0,y0],omega))
                end
            end
        end
    end
    return U
end


















@everywhere function in_Domain(a,b)
    if (a^2 + b^2 <= 1)
        return true
    end
    return false
end

@everywhere function scattering(omega,omegaP)
    return 1 / (2*pi)
end

@everywhere function quadrature(innerFunction,a,b,N)
    vec = range(a,stop=b,length=N)
    ddx = vec[2] - vec[1]
    answer = 0.0
    for i=2:length(vec)
        answer = answer + (ddx / 2) * (innerFunction(vec[i]) + innerFunction(vec[i-1]))
    end
    return answer
end

@everywhere function interpolant_intermediate(u,dx,dy,dz,x)
    M = dx
    N = dy
    WW = dz
    X = range(-1,stop=1,length=M);
    Y = range(-1,stop=1,length=N);
    W = range(0,stop=2*pi-2*pi/WW,length=WW);

    #x_L,x_H are the 1-based indices of points in u
    x_index = 1
    for i=1:size(X)[1]
        if X[i] > x[1]
            x_index= i
            break;
        end

    end
    x_L = x_index-1
    x_H = x_index
    x_L = max(x_L,1)
    x_H = min(x_H,M)

    y_index = 1
    for i=1:size(Y)[1]
        if Y[i] > x[2]
            y_index= i
            break;
        end
    end
    y_L = y_index-1
    y_H = y_index
    y_L = max(y_L,1)
    y_H = min(y_H,N)



    z_index = 1
    for i=1:size(W)[1]
        if W[i] > x[3]
            z_index= i
            break;
        end
    end
    z_L = z_index-1
    z_H = z_index
    z_L = max(z_L,1)
    z_H = min(z_H,WW)
    #(x_L, y_L, z_L)
    distance_1 = distance(x,[dx*(x_L-1),dy*(y_L-1),dz*(z_L - 1)])
    #(x_H, y_L, z_L)
    distance_2 = distance(x,[dx*(x_H-1),dy*(y_L-1),dz*(z_L - 1)])
    #(x_L, y_H, z_L)
    distance_3 = distance(x,[dx*(x_L-1),dy*(y_H-1),dz*(z_L - 1)])
    #(x_H, y_H, z_L)
    distance_4 = distance(x,[dx*(x_H-1),dy*(y_H-1),dz*(z_L - 1)])
    #(x_L, y_L, z_H)
    distance_5 = distance(x,[dx*(x_L-1),dy*(y_L-1),dz*(z_H - 1)])
    #(x_H, y_L, z_H)
    distance_6 = distance(x,[dx*(x_H-1),dy*(y_L-1),dz*(z_H - 1)])
    #(x_L, y_H, z_H)
    distance_7 = distance(x,[dx*(x_L-1),dy*(y_H-1),dz*(z_H - 1)])
    #(x_H, y_H, z_H)
    distance_8 = distance(x,[dx*(x_H-1),dy*(y_H-1),dz*(z_H - 1)])

    #If we're trying to approximate a point already in u, return it
    if (distance_1 == 0)
        return u[x_L,y_L,z_L]
    end
    if (distance_2 == 0)
        return u[x_H,y_L,z_L]
    end
    if (distance_3 == 0)
        return u[x_L,y_H,z_L]
    end
    if (distance_4 == 0)
        return u[x_H,y_H,z_L]
    end
    if (distance_5 == 0)
        return u[x_L,y_L,z_H]
    end
    if (distance_6 == 0)
        return u[x_H,y_L,z_H]
    end
    if (distance_7 == 0)
        return u[x_L,y_H,z_H]
    end
    if (distance_8 == 0)
        return u[x_H,y_H,z_H]
    end

    #NOTE: Even when leaving out the above if-statements, I was still getting very accurate answers. Commands such as:
    """
    if (distance_1 == 0)
        distance_1 = (*Big Number*)
    end
    """
    # Would return very accurate numbers, the larger "big number" (weight) was. I decided to forego that because there's no use calculating a distance we already have and it seemed pretty accurate.


    distance_1 = 1/distance_1
    distance_2 = 1/distance_2
    distance_3 = 1/distance_3
    distance_4 = 1/distance_4
    distance_5 = 1/distance_5
    distance_6 = 1/distance_6
    distance_7 = 1/distance_7
    distance_8 = 1/distance_8


    #From distances, calculate weights
    sum = distance_1 + distance_2 + distance_3 + distance_4 + distance_5 + distance_6 + distance_7 + distance_8

    w_1 = distance_1/sum
    w_2 = distance_2/sum
    w_3 = distance_3/sum
    w_4 = distance_4/sum
    w_5 = distance_5/sum
    w_6 = distance_6/sum
    w_7 = distance_7/sum
    w_8 = distance_8/sum
    answer = w_1*u[x_L,y_L,z_L] + w_2*u[x_H,y_L,z_L] + w_3*u[x_L,y_H,z_L] + w_4*u[x_H,y_H,z_L] + w_5*u[x_L,y_L,z_H] + w_6*u[x_H,y_L,z_H] + w_7*u[x_L,y_H,z_H] + w_8*u[x_H,y_H,z_H]
    return answer
end
# Calculate distance between two 3-dimensional points / vectors.
@everywhere function distance(x1,x2)
    return sqrt((x1[1] - x2[1])^2 + (x1[2] - x2[2])^2 + (x1[3] - x2[3])^2)
end
