using Distributions, Random, Plots, ProgressMeter, LinearAlgebra, Random
using Bijectors
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


##################
# building blacks of realnvp
################
function lrelu_layer(xdims::Int; hdims::Int=20)
    nn = Chain(Flux.Dense(xdims, hdims, leakyrelu), Flux.Dense(hdims, hdims, leakyrelu), Flux.Dense(hdims, xdims))
    return nn
end
@functor lrelu_layer
"""
coupling function for RealNVP "(θ function in Eq(1) of http://proceedings.mlr.press/v118/fjelde20a/fjelde20a.pdf)"
"""
function affine_coupling_layer(shifting_layer, scaling_layer, dims, masking_idx)
    Bijectors.Coupling(θ -> Bijectors.Shift(shifting_layer(θ)) ∘ Bijectors.Scale(scaling_layer(θ)), Bijectors.PartitionMask(dims, masking_idx))
end


function RealNVP_layers(q0, nlayers, d; hdims=20)
    xdims = Int(d/2)
    # println(xdims)
    scaling_layers = [ lrelu_layer(xdims; hdims = hdims) for i in 1:nlayers ]
    shifting_layers = [ lrelu_layer(xdims; hdims = hdims) for i in 1:nlayers ]
    ps = Flux.params(shifting_layers[1], scaling_layers[1]) 
    Layers = affine_coupling_layer(shifting_layers[1], scaling_layers[1], d, xdims+1:d)
    # number of affine_coupling_layers with alternating masking scheme
    for i in 2:nlayers
        Flux.params!(ps, (shifting_layers[i], scaling_layers[i]))
        Layers = Layers ∘ affine_coupling_layer(shifting_layers[i], scaling_layers[i], d, (i%2)*xdims+1:(1 + i%2)*xdims) 
    end
    flow = Bijectors.transformed(q0, Layers)
    return flow, Layers, ps
end


"""
training loop for realnvp:
    stochastic gradient asenct for ELBO 
"""
function train_rnvp(q0, logp, logq, d::Int, niters::Int; 
                nlayers = 5, hdims = 20,  
                elbo_size::Int = 10, optimizer = Flux.ADAM(1e-3), kwargs...)
    flow, Layers, ps = RealNVP_layers(q0, nlayers, d; hdims = hdims)
    #define loss
    loss = () -> begin 
        elbo = nf_ELBO(flow, logp, logq; elbo_size = elbo_size)
        return -elbo
    end
    loss_log, _ = vi_train!(niters, loss, ps, optimizer; logging_ps= false, kwargs...)
    trained_ps = [[copy(p) for p in ps]]
    return flow, trained_ps, -loss_log
end




####################3
# flow training
###################
# initialize planar flow of 10 layers
q0 = MvNormal(zeros(Float32, 2), I) # refrence distirbution
logq(x) =  -0.5*d*log(2π)  - 0.5*sum(abs2, x) # reference distribution of RealNVP
# construct realnvp
nlayers = 20
flow, Layers, ps = RealNVP_layers(q0, nlayers, 2; hdims = 5);

# ELBO before training
el_ut = nf_ELBO(flow, logp, logq; elbo_size = 1000)
# collecting 1000 samples from untrained nf
T_ut = rand(flow, 1000)

# train nf
niter = 50_000
flow, _, el = train_rnvp(q0, logp, logq, 2, niter; nlayers = nlayers, hdims = 5, elbo_size = 10)

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