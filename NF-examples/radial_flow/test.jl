using Distributions, Random, Plots, ProgressMeter, LinearAlgebra, Random
using StatsBase, SpecialFunctions, Parameters, Bijectors
using ProgressMeter, Flux, Bijectors, Zygote
include("../utils/train.jl") # loading utility functions for optimization loops
include("../utils/elbo.jl") # loading utility functions for elbo estimation


d = 2 # dimension
b = 0.1 # curvature
Z = sqrt(100 * (2*π)^d) # normalizing constant
C = Matrix(Diagonal(vcat(100, ones(d-1))))
C_inv = Matrix(Diagonal(vcat(1/100, ones(d-1))))
ϕ_inv(y) = [y[1], y[2] - b*y[1]^2 + 100*b]
logp(x) = -log(Z) - 0.5 * ϕ_inv(x)' * C_inv * ϕ_inv(x) # log pdf of the target distribution
# ∇logp(x) = -[1/100 * x[1] + (x[2]-b*x[1]^2+100*b)*(-2*b*x[1]), x[2]-b*x[1]^2+100*b]
# ∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)



####################3
# flow training
###################
# initialize planar flow of 10 layers
nlayers = 20
F = ∘([RadialLayer(d) for i in 1:nlayers]...)
q0 = MvNormal(zeros(Float32, 2), I) # refrence distirbution
logq(x) =  -0.5*d*log(2π)  - 0.5*sum(abs2, x) # reference distribution of RealNVP
flow = transformed(q0, F)

# ELBO before training
el_ut = nf_ELBO(flow, logp, logq; elbo_size = 1000)
# collecting 1000 samples from untrained nf
T_ut = rand(flow, 1000)

# train nf
niter = 50_000
_, el = nf(flow, logp, logq, niter; elbo_size = 10)

# ELBO after training
el_nf = nf_ELBO(flow, logp, logq; elbo_size = 1000)
# collecting 1000 samples from trained nf
T_nf = rand(flow, 1000)



####################3
# plotting
###################
# check ELBO value over training iterations
plot(el, label = "ELBO", legendfontsize= 15)
hline!([el_ut], label = "ELBO (untrained)", lw = 2, linestyle = :dash, legendfontsize= 15, legend=:best)
hline!([el_nf], label = "ELBO (trained)", lw = 2, linestyle = :dash, legendfontsize= 15, legend=:best)

# check sample quality difference between untrained and trained nf
x = -20:.1:20
y = -15:.1:30
pdf_target = (x, y) -> exp(logp([x,y]))        
p1 = contour(x, y, pdf_target, colorbar = false, xlim = (x[1], x[end]), ylim = (y[1], y[end]))
scatter!(T_ut[1, :], T_ut[2, :], label = "NF samples (untrained)", color = 1, legendfontsize= 15, legend=:top)
p2 = contour(x, y, pdf_target, colorbar = false, xlim = (x[1], x[end]), ylim = (y[1], y[end]))
scatter!(T_nf[1, :], T_nf[2, :], label = "NF samples (trained)", color = 1, legendfontsize= 15, legend=:top)
plot(p1, p2, layout = (1,2), size = (1000, 400))