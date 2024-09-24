include("_UnivariateKDE.jl")

Base.@kwdef struct MultivariateKDE{K<:Distribution{Univariate,Continuous}, D<:AbstractMatrix, T<:AbstractMatrix} <: Distribution{Multivariate, Continuous}
    kernel    :: K
    points    :: D
    bandwidth :: Cholesky{eltype(T),T}
end

function MultivariateKDE{K}(x::AbstractMatrix{<:Real}, bandwidth::Cholesky) where K<:Distribution{Univariate,Continuous}
    return MultivariateKDE{K,typeof(x),typeof(bandwidth)}(
        kernel = __kernel(K),
        points = bandwidth.L\x',
        bandwidth = bandwidth
    )
end

function MultivariateKDE{K}(x::AbstractMatrix{<:Real}, bandwidth::AbstractMatrix) where K<:Distribution{Univariate,Continuous}
    return MultivariateKDE{K}(x, cholesky(bandwidth))
end

MultivariateKDE(x::V, h::T) where {V,T} = MultivariateKDE{Normal{Float64}}(x, h)

Base.length(kde::MultivariateKDE) = size(kde.points, 2)
dimension(kde::MultivariateKDE) = size(kde.points, 1)

StandardWeight(kde::MultivariateKDE, x::AbstractVector) = StandardWeight(
    kernel = kde.kernel,
    b = 1/(length(kde)*det(kde.bandwidth.L)),
    z = kde.bandwidth.L\x
)

function Distributions.rand(rng::Distributions.AbstractRNG, kde::MultivariateKDE)
    ii = ceil(Int64, length(kde)*rand())
    z  = rand(kde.kernel, dimension(kde))
    return kde.bandwidth.L*((@view kde.points[:,ii]), .+ z)
end

function Distributions.pdf(kde::MultivariateKDE, x::AbstractVector)
    s = StandardWeight(kde, x)
    return sum(s, eachcol(kde.points))
end
Distributions.logpdf(kde::MultivariateKDE, x::AbstractVector) = log(pdf(kde, x))

function kernel_weights(kde::MultivariateKDE, x::AbstractVector)
    s = StandardWeight(kde, x)
    return map(s, eachcol(kde.points))
end

function kernel_weight(kde::MultivariateKDE, x, ii::Integer)
    s = StandardWeight(kde, x)
    return s(kde.points[ii])
end

