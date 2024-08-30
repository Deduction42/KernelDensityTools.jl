module KernelDensityTools
    include("_KernelRegression.jl")
    export 
        AbstractKernelDensity,
        UnivariateKDE,
        MultivariateKDE,
        KernelGrid,
        KernelInterp,
        silvermans_rule,
        predict,
        pdf
end
