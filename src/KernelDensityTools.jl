module KernelDensityTools
    include("_KernelRegression.jl")
    export 
        AbstractKernelDensity,
        UnivariateKDE,
        MultivariateKDE,
        KernelRegGrid,
        KernelRegInterp,
        silvermans_rule,
        predict,
        pdf
end
