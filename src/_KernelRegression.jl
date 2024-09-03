include("_KernelDensity.jl")

# The nadaraya-watson estimator uses kernel density weights 
using SparseArrays
using Interpolations
using OffsetArrays
import NaNMath

# ======================================================================================================
# Nadaraya Watson estimator that stores weights for a specific set of desired sample points
# Let us say that we have related daya (x,y) and we want a set of estimates for sampled x-points xs
# The estimator takes samples points and computes the weights for x
# These weights can be applied to any y that maps to x in order to obtain predictions at xs
# ======================================================================================================
struct SampleKernelWeights{T<:AbstractVector}
    samples :: T
    weights :: Vector{SparseVector{Float64,Int64}}
end
Base.length(θ::SampleKernelWeights) = length(θ.samples)

function SampleKernelWeights(n::Integer, x::AbstractVector; scale=1.0)
    return SampleKernelWeights(samplegrid(x, n), x, scale=scale)
end

function SampleKernelWeights(samples::AbstractVector, x::AbstractVector{<:Real}; scale=1.0)
    Ns = length(samples)
    Nx = length(x)
    weights = [ spzeros(Float64, Int64, Nx) for ii in 1:Ns ]

    KDE = UnivariateKDE{Normal}(x, scale*silvermans_rule(x))

    Threads.@threads for ii in 1:Ns
        w = replace!(kernel_weights(KDE, samples[ii]), NaN=>0) #Obtain kernel weights (replace NaNs with 0)
        w .= w./sum(w) #Stanardize by the sum for a Nadaraya-Watson estimator
        ind = w .> 0.5*eps(Float64) #Eliminate weights that are smaller than machine precision
        weights[ii][ind] .= w[ind] 
    end

    return SampleKernelWeights{typeof(samples)}(samples, weights)
end

# ======================================================================================================
# Grid-based representation of the 1-d estimators for export and to feed interpolation
# ======================================================================================================
struct KernelRegGrid{T<:Real}
    s :: LinRange{T, Int64}
    y :: Vector{Float64}
end

function KernelRegGrid(y::AbstractVector, θ::SampleKernelWeights{T}) where T <: AbstractRange
    return KernelRegGrid(θ.samples, smooth(y, θ))
end

# ======================================================================================================
# Interpolations of the NadarayaWatsonGrid to accelerate evaluation
# ======================================================================================================

#We only need to use one type of interpolation, to change the type, run the constructor function on NadarayaWatsonInterp(G::NadarayaWatsonGrid) and paste the output type here
const CubicSplineInterp = Interpolations.BSplineInterpolation{T, 1, OffsetArrays.OffsetVector{T, Vector{T}}, BSpline{Cubic{Flat{OnGrid}}}, Tuple{Base.OneTo{Int64}}} where T <: Number

struct KernelRegInterp{T<:Real}
    interp :: CubicSplineInterp{T}
    bounds :: Tuple{Float64, Float64}
end

function KernelRegInterp(G::KernelRegGrid)
    bSpline = BSpline(Cubic(Flat(OnGrid())))
    interp  = interpolate(G.y, bSpline)
    bounds  = extrema(G.s)
    return KernelRegInterp{eltype(G.y)}(interp, bounds)
end

function KernelRegInterp(θ::SampleKernelWeights{T}, y::AbstractVector{<:Real}) where T <: AbstractRange
    G = KernelRegGrid(y, θ)
    return KernelRegInterp(G)
end

#This provides an additional step of using a linear regression correction based on x to reduce attenuation bias
function KernelRegInterp(θ::SampleKernelWeights{T}, y::AbstractVector{<:Real}, x::AbstractVector{<:Real}) where T <: AbstractRange
    G  = linear_kernel_correction(θ, y, x)
    return KernelRegInterp(G)
end

# ======================================================================================================
# Produces a function grid of Nadaraya-Watson estimators with an additional linear fiting step
# Linear fitting step reduces flattening bias from estimator
# ======================================================================================================
function linear_kernel_correction(θ::SampleKernelWeights{T}, y::AbstractVector{<:Real}, x::AbstractVector{<:Real}) where T <: AbstractRange
    fx  = Ref(KernelRegInterp(θ, y))
    ys  = predict.(fx, x)
    ind = (isfinite.(ys) .& isfinite.(y))
    keepat!(ys, ind)

    if allequal(ys) #No point in linear fitting if all ys0 are equal
        yh = predict.(fx, θ.samples)
        return KernelRegGrid(θ.samples, yh)

    else #Scale estimator to account for flattening bias
        β  = dot(ys, y[ind])/dot(ys, ys)
        yh = predict.(fx, θ.samples).*β
        return KernelRegGrid(θ.samples, yh)
    end
end


# ======================================================================================================
# Smoothing and estimation functions
# ======================================================================================================
function predict(θ::AbstractVector{<:KernelRegInterp}, X::AbstractMatrix{<:Real})
    (N, P) = size(X)
    
    if length(θ) != P
        throw(DimensionMismatch("Prediction was given $(length(θ)) models which does not match the number of series $(P)."))
    end
    
    T  = typeof( predict(first(θ), first(X)) )
    Yh = zeros(T, (N,P))

    for (cy, cx, im) in zip(axes(Yh,2), axes(X,2), eachindex(θ))
        m = θ[im]
        for (ry, rx) in zip(axes(Yh,1), axes(X,1))
            Yh[ry, cy] = predict(m, X[rx, cx])
        end
    end

    return Yh
end


function predict(θ::KernelRegInterp, x::T) where T <: Real
    if isnan(x) #Skip this if x is NaN
        return x
    end

    #Calculate standard value 
    z  = (x-θ.bounds[1])/(θ.bounds[2]-θ.bounds[1])
    z  = ifelse(isnan(z), T(0.5), z) #Rare case wher bounds[1]==bounds[2] and x is in bounds

    #ii is the interpolation index between 1 and the length of the dataset, see Interpolations.jl
    ii = clamp(z,0,1)*(length(θ.interp)-1) + 1
    return θ.interp(ii)
end

# Smooth out "y" values for all the sample points in θ
function smooth(y::AbstractVector, θ::SampleKernelWeights)    
    return [ smooth(y, w) for w in θ.weights ]
end

# This is essentially a dot product that skips NaNs in y (because we already got rid of them in w)
# see https://github.com/JuliaSparse/SparseArrays.jl/blob/main/src/sparsevector.jl
function smooth(y::AbstractVector{Ty}, w::SparseVector{Tw}) where {Ty<:Number, Tw<:Number}
    (length(y) == length(w)) || throw(DimensionMismatch())
    Base.require_one_based_indexing(y)

    nzind = SparseArrays.nonzeroinds(w)
    nzval = SparseArrays.nonzeros(w)
    s = dot(zero(Ty), zero(Tw))
    ρ = zero(Tw)

    for ii = 1:length(nzind)
        yi = y[nzind[ii]]
        wi = nzval[ii]
        nani = isnan(yi) 

        s += ifelse(nani, zero(Ty), dot(yi,wi))
        ρ += ifelse(nani, wi, zero(Tw))
    end

    return s/(1-ρ)
end


#Create a LinRange for maximum and minimum values of x, ignoring NaNs
function samplegrid(x::AbstractVector, n::Integer)
    Δx = NaNMath.extrema(x)
    return LinRange(Δx[1], Δx[2], n)
end

