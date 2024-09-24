using LinearAlgebra
using Statistics
using Distributions
using NaNStatistics
using StaticArrays

@kwdef struct StandardWeight{K<:Distribution{Univariate,Continuous}, B<:Real, Z}
    kernel :: K
    b :: B
    z :: Z
end

function (s::StandardWeight{<:Any,<:Any,<:Real})(point::Real) 
    return s.b*pdf(s.kernel, s.z-point)
end

function (s::StandardWeight{<:Any,<:Any,<:AbstractVector})(point::AbstractVector)
    return s.b*prod(Base.Fix1(pdf, s.kernel), s.z-point)
end

#Canonical list of supported kernels
KERNELS = (Normal, Epanechnikov, Triweight, Biweight, Uniform, Logistic, Cosine)

__kernel(d::T) where T = __kernel(T)

for Dist in KERNELS
    if Dist == Uniform
        @eval begin
            __kernel(::Type{$Dist{T}}) where T = $Dist{T}(-T(0.5), T(0.5))
            __kernel(::Type{$Dist}) = __kernel($Dist{Float64})
        end
    else
        @eval begin 
            __kernel(::Type{$Dist{T}}) where T = $Dist{T}(zero(T), one(T))
            __kernel(::Type{$Dist}) = __kernel($Dist{Float64})
        end
    end
end


silvermans_rule(n::Integer) = (4/(n*3))^(1/5)
silvermans_rule(x::AbstractVector) = nanstd(x)*silvermans_rule(length(x))

silvermans_rule(n::Integer, d::Integer) = (4/(n*(d+2)))^(2/(d+4))
silvermans_rule(x::AbstractMatrix) = nancov(x)*silvermans_rule(size(x,1), size(x,2))