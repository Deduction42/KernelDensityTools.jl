using LinearAlgebra
using Statistics
using Distributions
using NaNStatistics

Base.@kwdef struct UnivariateKDE{F<:Function} <: Distribution{Univariate, Continuous}
    points    :: Vector{Float64}
    h         :: Float64
    scale     :: Float64
    kernel    :: F
end

Base.@kwdef struct MultivariateKDE{F<:Function, M<:AbstractMatrix} <: Distribution{Univariate, Continuous}
    points    :: Matrix{Float64}
    h         :: M
    scale     :: Float64
    kernel    :: F
end

const AbstractKDE = Union{UnivariateKDE, MultivariateKDE}


Base.length(D::AbstractKDE) = size(D.points, 1)

function UnivariateKDE(x::AbstractVector{<:Real}, h::Real; kernel=normal_kernel)
    return UnivariateKDE(
        points = Vector(x),
        h      = h,
        scale  = 1/(h*length(x)),
        kernel = kernel
    )
end

function MultivariateKDE(x::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}; kernel=mv_normal_kernel)
    return MultivariateKDE(
        points = Matrix(x),
        h      = h,
        scale  = 1/(det(h)*size(x,1)),
        kernel = kernel
    )
end

function Distributions.pdf(D::AbstractKDE, x::Real)
    return sum(kernel_weights(D,x))
end

function Distributions.rand(rng::Distributions.AbstractRNG, D::UnivariateKDE)
    ii = ceil(Int64, length(D)*rand())
    return rand(Normal(D.points[ii], D.h))
end



function kernel_weights(D::AbstractKDE, x::Union{T, AbstractVector{T}}) where T <: Real
    w = zeros(promote_type(T, Float64), length(D))
    for ii in 1:length(D)
        w[ii] = kernel_weight(D, x, ii)
    end
    return w
end

function kernel_weight(D::UnivariateKDE, x::Real, ii::Integer)
    return D.scale * D.kernel( (x-D.points[ii])/D.h )
end

function kernel_weight(D::MultivariateKDE, x::AbstractVector{<:Real}, ii::Integer)
    return @views D.scale * D.kernel( D.h\(x-D.points[ii,:]) )
end

function silvermans_rule(x::AbstractVector)
    (n,d) = (length(x),1)
    h = (4/(n*(d+2)))^(1/(d+4))
    return h*nanstd(x)
end

function silvermans_rule(x::AbstractMatrix; diagonal=true)
    (n,d) = size(x)
    h = (4/(n*(d+2)))^(1/(d+4))
    S = nancov(x)
    
    if diagonal
        return h*Diagonal(sqrt.(diag(S)))
    else
        return h*cholesky(S).L
    end
end


const log2π = log(2*π)

function normal_kernel(x::Real)
    return exp(-0.5*(x^2 + log2π))
end

function mv_normal_kernel(x::AbstractVector{<:Real})
    d = length(x)
    return exp(-0.5*(dot(x,x) + d*log2π))
end