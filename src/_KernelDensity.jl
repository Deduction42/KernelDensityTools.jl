using LinearAlgebra
using Statistics
using Distributions
using NaNStatistics
using StaticArrays

Base.@kwdef struct UnivariateKDE{D<:Distribution} <: Distribution{Univariate, Continuous}
    points      :: Vector{Float64}
    kernel      :: D
end

Base.@kwdef struct MultivariateKDE{N, D<:Distribution} <: Distribution{Multivariate, Continuous}
    points      :: Vector{SVector{N, Float64}}
    kernel      :: D
end

const AbstractKDE = Union{UnivariateKDE, MultivariateKDE}


Base.length(D::AbstractKDE) = size(D.points, 1)

function UnivariateKDE{D}(x::AbstractVector{<:Real}, h::Real) where D <: Distribution
    return UnivariateKDE(
        points = Vector(x),
        kernel = scaled_kernel(D, h)
    )
end
UnivariateKDE(x::AbstractVector{<:Real}, h::Real) = UnivariateKDE{Normal}(x, h)

function MultivariateKDE{N,D}(X::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real}) where {N, D<:Distribution}
    return MultivariateKDE{N,D}(
        points = [SVector{N}(r) for r in eachrow(X)],
        kernel = scaled_kernel(D, SMatrix{N,N}(h))
    )
end
MultivariateKDE{N}(x::Vector{SVector{N, <:Real}}, h::AbstractMatrix{<:Real}) where N = MultivariateKDE{N, MvNormal}(x, h)

function Distributions.pdf(kde::AbstractKDE, x::Real)
    return sum(kernel_weights(kde,x))
end

function Distributions.rand(rng::Distributions.AbstractRNG, kde::UnivariateKDE)
    ii = ceil(Int64, length(kde)*rand())
    return kde.points[ii] + rand(kde.kernel)
end

function kernel_weights(kde::AbstractKDE, x)
    n = length(kde)
    return [pdf(kde.kernel, x-p)/n for p in kde.points]
end

function kernel_weight(kde::UnivariateKDE, x, ii::Integer)
    return pdf(kde.kernel, x-kde.points[ii])/length(kde)
end

function silvermans_rule(x::AbstractVector)
    (n, d) = (length(x),1)
    h = (4/(n*(d+2)))^(1/(d+4))
    return h*nanstd(x)
end

function silvermans_rule(x::AbstractMatrix)
    (n, d) = size(x)
    h² = (4/(n*(d+2)))^(2/(d+4))  
    return h²*nancov(x)
end

"""
Produce a scaled kernel with mean 0 and scale according to bandwidth
"""
scaled_kernel(::Type{D}, h::Real) where D <: Distribution{Univariate, Continuous} = D(0, h)
scaled_kernel(::Type{Uniform}, h::Real) = Uniform(-0.5*h, 0.5*h)
scaled_kernel(::Type{MvNormal}, h::SMatrix{N,N}) where N = MvNormal(zero(SVector{N, Float64}), h)
