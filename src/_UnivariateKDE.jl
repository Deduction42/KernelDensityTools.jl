include("_StandardWeight.jl")

Base.@kwdef struct UnivariateKDE{K<:Distribution{Univariate,Continuous}, D<:AbstractVector, T<:Real} <: Distribution{Univariate, Continuous}
    kernel    :: K
    points    :: D
    bandwidth :: T
end

function UnivariateKDE{K}(x::AbstractVector{<:Real}, bandwidth::Real) where K<:Distribution{Univariate,Continuous}
    return UnivariateKDE{K,typeof(x),typeof(bandwidth)}(
        kernel = __kernel(K),
        points = sort!(x/bandwidth),
        bandwidth = bandwidth
    )
end
UnivariateKDE(x::V, h::T) where {V,T} = UnivariateKDE{Normal{Float64}}(x, h)

Base.length(kde::UnivariateKDE) = length(kde.points)

StandardWeight(kde::UnivariateKDE, x::Real) = StandardWeight(
    kernel = kde.kernel,
    b = 1/(length(kde)*kde.bandwidth), 
    z = x/kde.bandwidth
)

function Distributions.rand(rng::Distributions.AbstractRNG, kde::UnivariateKDE)
    ii = ceil(Int64, length(kde)*rand())
    return kde.bandwidth*(kde.points[ii] + rand(kde.kernel))
end

function Distributions.pdf(kde::UnivariateKDE, x::Real)
    s = StandardWeight(kde, x)
    return sum(s, kde.points)
end

function kernel_weights(kde::UnivariateKDE, x::Real)
    s = StandardWeight(kde, x)
    return map(s, kde.points)
end

function kernel_weight(kde::UnivariateKDE, x, ii::Integer)
    s = StandardWeight(kde, x)
    return s(kde.points[ii])
end


