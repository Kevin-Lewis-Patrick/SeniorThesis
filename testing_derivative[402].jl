using SparseArrays,LinearAlgebra, Profile, PyPlot

function li(x,y,z,size)::Int64
    width = size[1]
    height = size[2]
    return x + (y-1) * width + (z-1) * width * height
end

function rli(index,size)::Array{Int64,2}
    answer = zeros((3,1))
    width = size[1]
    height = size[2]
    depth = size[3]
    answer[1] = mod(index-1,width)+1
    answer[2] = mod(floor((index-1) / width),height) + 1
    answer[3] = floor((index-1)/(height * width))+1
    convert(Array{Int64},answer)
    return answer
end

function g(x,y,omega)
    return x^2 + x*y - omega/(x+1)
end

function gx(x,y,omega)
    return 2*x + y + omega / ((x+1)^2)
end

# Create Dx
function create_Dx(U,dn)
    sizeU = size(U)[1] * size(U)[2] * size(U)[3]
    Dx = spzeros(sizeU,sizeU)
    for i=1:(size(U)[1] * size(U)[2] * size(U)[3])
        ix = rli(i,size(U)) #ix returns [x,y,z] of U according to what row of Ux you're on
        if (ix[1] < size(U)[1])
            ix_P = trunc(Int,li(ix[1]+1,ix[2],ix[3],size(U)))
            Dx[i,ix_P] = 1 / dn
            Dx[i,i] = -1 / dn
        else #ix = size(X)[1]

            ix_L = trunc(Int,li(ix[1]-1,ix[2],ix[3],size(U)))
            Dx[i,ix_L] = -1 / dn
            Dx[i,i] = 1 / dn

        end
    end
    return Dx
end

function create_Dy(U,dn)
    sizeU = size(U)[1] * size(U)[2] * size(U)[3]
    Dx = spzeros(sizeU,sizeU)
    for i=1:(size(U)[1] * size(U)[2] * size(U)[3])
        ix = rli(i,size(U)) #ix returns [x,y,z] of U according to what row of Ux you're on

        if (ix[2] < size(U)[1])
            ix_P = trunc(Int,li(ix[1],ix[2]+1,ix[3],size(U)))
            Dx[i,ix_P] = 1 / dn
            Dx[i,i] = -1 / dn
        else

            ix_L = trunc(Int,li(ix[1],ix[2]-1,ix[3],size(U)))
            Dx[i,ix_L] = -1 / dn
            Dx[i,i] = 1 / dn

        end
    end
    return Dx
end

test_rest = 0
if (test_rest==1)
    dx = 1/40
    dy = dx
    dw = dx

    X = range(0,stop=1,length=floor(Int64, 1/dx));
    Y = range(0,stop=1,length=floor(Int64, 1/dy));
    W = range(0,stop=1,length=floor(Int64, 1/dw));

    U = zeros(size(X)[1],size(Y)[1],size(W)[1])
    Ux = zeros(size(X)[1],size(Y)[1],size(W)[1])

    Dx = create_Dx_Dy(U,dx)



    println("size of U: ",size(U))
    for i=1:size(U)[1]
        for j=1:size(U)[2]
            for k=1:size(U)[3]
                U[i,j,k] = g(X[i],Y[j],W[k])
                Ux[i,j,k] = gx(X[i],Y[j],W[k])
            end
        end
    end

    U_2 = zeros(size(Dx)[1],1)
    Ux_2 = zeros(size(Dx)[1],1)
    for i=1:size(Dx)[1]
        ans = rli(i,size(U))
        x = floor(Int64,ans[1])
        y = floor(Int64,ans[2])
        z = floor(Int64,ans[3])
        U_2[i] = U[x,y,z]
        Ux_2[i] = Ux[x,y,z]
    end
    println("Dx is: ")
    #display(Dx)

    println("size of U_2: ", size(U_2))
    println("size of Ux_2: ", size(Ux_2))


    A = Dx * U_2
    #println(A)
    println("NORM: ", norm(A-Ux_2,Inf))

    #println("A is: ")
    #display(A)
    #println("124532145")
    #println("Ux_2 is: ")
    #display(Ux_2)
end
