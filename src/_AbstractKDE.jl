include("_MultivariateKDE.jl")

const AbstractKDE = Union{UnivariateKDE, MultivariateKDE}

using LinearAlgebra
using SpecialFunctions

#========================================================================================
ToDo:
(1) Input is now a cholesky factorization
(2) Expected value of samples from a Normal distribution, given a volume v should now be
        n0 = n*V*(1/2)^(d/2)  (verify this shortcut formula)
========================================================================================#

function multimode_bandwidth(kde::MultivariateKDE)
    #Obtain Silverman's BWM
    Z = kde.points
    (d, n) = size(Z)
    s = silvermans_rule(n,d)

    #Standardized values of X according to bandwidth matrix
    Z = (H.L*sqrt(s))\X'
  
    #Volume covered by optimal bandwidth matrix for N(0,1)
    V = ( pi^(d/2) / gamma(d/2+1) )*( sqrt(s) )^d
  
    #Expected number of other data points covered by that volume
    K(S) = sqrt( (2*pi)^(size(S,1)) * det(S) )
    S  = Diagonal(ones(d))
    n0 = n*V*(0.5)^(0.5*d)
  
    #Calculate the actual average number of data points covered
    # nx = N*V*P(x) (number of samples in a sphere around x)
    #E(nx) = ∫N*V*P(x)*P(x) dx
    #      = N*V ∫P(x)^2 dx
    #      = N*V ∫ 1/K(S)^2 exp(-1/2 x'*(S/2)^(-1)*x) dx
    #      = N*V 1/K(S)^2 ∫ exp(-1/2 x'*(S/2)^(-1)*x) dx
    #E(nx) = N*V* K(S/2)/K(S)^2
  
    
  
    #Calculate the squared euclidian distances for all points and place in a matrix
    D0 = __column_distances(Z)
    D1 = deepcopy(D0)
  
    #Determine the average number of data points captured by each bandwidth
    n1 = (1/n)*find_previous(1.0, D2) # = sum(D2.< 1.0)
  
    #Iteration record
    i = 0
  
    #Rescale the distances and see how many points are covered by bandwidth
    #New distances are scaled by s, iterate s estimates until n1=n0
    #Old algorithm used only one iteration which did not converge yet
    while ( abs(log(n0/n1)) > 0.01 ) & (i <=20 )
      i = i+1
      s = s*(n0/n1)^(1/d)
      #n1 = (1/n)*sum( D2/(s^2) .<= 1 )
      n1 = (1/n)*find_previous(s^2, D2)
    end
  
    return s
end

function __column_distances(Z::AbstractMatrix{T}) where T
    d = zeros(T, size(Z,2), size(Z,2))
    coldist(i::Integer, j::Integer) = __column_distance(Z, i, j)

    for (id, iz) in enumerate(axes(Z,2))
        d[:,id] .= coldist.(iz, axes(Z,2))
    end
    
    return d
end

function __column_distance(Z::AbstractMatrix, i::Integer, j::Integer)
    return sum(x->abs2(x[1]-x[2]), zip(view(Z, :, i), view(Z, :, j)))
end

X = randn(5, 1000)
@time D0 = __column_distances(X)
