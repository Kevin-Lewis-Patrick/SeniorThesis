include("testing_derivative.jl")
using ProgressMeter
function forcing(x, omega)
    if (x[1]^2 + x[2]^2 < 0.25^2)
        return 1
    else
        return 0
    end
end

function scattering(omega,omegaP)
    return 1/(2*pi)
end

nx = 80
ny = nx
nw = 28
X = range(-1,stop = 1,    length = nx);
Y = range(-1,stop = 1,    length = ny);
W = range(0, stop = 2*pi-(2*pi/nw), length = nw);



mu_a = 2

delx = X[2]-X[1]
dely = Y[2]-Y[1]


triplet_Size = size(X)[1] * size(Y)[1] * size(W)[1]
U =      zeros(size(X)[1],size(Y)[1],size(W)[1])
Dx =     spzeros(triplet_Size, triplet_Size)
Dy =     spzeros(triplet_Size, triplet_Size)
Wx =     spzeros(triplet_Size, triplet_Size)
Wy =     spzeros(triplet_Size, triplet_Size)
sigmaT = spzeros(triplet_Size, triplet_Size)
inte =   spzeros(triplet_Size, triplet_Size)

Dx = create_Dx(U,delx)
Dy = create_Dy(U,dely)
println("created Dx, Dy")

# Initialize f_mat
f_mat = zeros((size(U)[1] * size(U)[2] * size(U)[3],1))
println("size(f_mat): ",size(f_mat))
for w= 1:size(U)[3], j=1:size(U)[2], i=1:size(U)[1]
    f_mat[li(i,j,w,size(U))] = forcing([X[i], Y[j]], W[w])
end

println("creating sigma_t")
# Create sigmaT, MNW x MNW matrix.
for i=1:size(sigmaT)[1]
    sigmaT[i,i] = mu_a
end

# Combine this with for 2nd above for loop for efficiency?
for i=1:size(U)[3]
    omega = W[i]

    for j=1:(size(U)[1]*size(U)[2])
        index = size(U)[2]*size(U)[1]*(i-1) + j
        Wx[index, index] = cos(omega)
        Wy[index, index] = sin(omega)
    end
end

mu_s = 0.1
println("triplet_Size = ",triplet_Size)
@showprogress "Computing . . ." for i=1:triplet_Size
    ans = rli(i,size(U))
    sizeW = size(W)[1]

    for j=1:size(U)[3]
        inte[i,li(ans[1],ans[2],j,size(U))] = mu_s * scattering(W[ans[3]],W[j])/sizeW
    end
end










A = Dx*Wx + Dy*Wy + sigmaT - inte

#Now, add (1000 x 1000) to the end of A, and 1000 to end of f
y = size(A)[1]
A     = [A;     one(spzeros(size(A)[1],size(A)[1]))]
f_mat = [f_mat; zeros(y,1)]

println("size(A): ",size(A))
println("size(f_mat): ",size(f_mat))
println("test: ",y)










A = factorize(A)

u_Sol = A \ f_mat

for i=1:size(u_Sol)[1]
    triplet = rli(i,size(U)) #fix casted output of rli eventually
    x = floor(Int64,triplet[1])
    y = floor(Int64,triplet[2])
    z = floor(Int64,triplet[3])
    U[x,y,z] = u_Sol[i]
end


toPlot = zeros(size(U)[1],size(U)[2])
for i=1:size(toPlot)[1]
    for j=1:size(toPlot)[2]
        if (X[i]^2 + Y[j]^2 < 1)
            toPlot[i,j] = sum(U[i,j,:])
        else
            toPlot[i,j]=0
        end
    end
end





#for w=1:size(W)[1]
#    figure();
#    print("Figure ",w," is angle: ",cos(W[w]),", ",sin(W[w]))
#    pcolormesh(X,Y,U[:,:,w])
#    colorbar()
#end
 pcolormesh(X,Y,toPlot[:,:])
 colorbar()

# if Ux is
println("size of Ux = ",Base.summarysize(Dx) / 1e9,", gigabytes.")
