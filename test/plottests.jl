using Revise
using Plots
using Distributions
using KernelDensityTools

N = 1500

d0 = Normal(3.2, 1.2)
x0 = rand(d0, N)
d1 = UnivariateKDE(x0, silvermans_rule(x0))
x1 = rand(d1, N)
d2 = UnivariateKDE(x1, silvermans_rule(x1))

vx = LinRange(minimum(x0), maximum(x0), 500)

ax = plot(vx, pdf.(d0, vx))
plot!(ax, vx, pdf.(d1, vx))
plot!(ax, vx, pdf.(d2, vx))
