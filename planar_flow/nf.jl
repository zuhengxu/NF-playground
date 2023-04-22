using Distributions, LinearAlgebra, Plots
using Bijectors
using Bijectors: PlanarLayer 
using Flux, Zygote
using Zygote: Params, pullback
using Optimisers
using ProgressMeter
using Plots

using Random
Random.seed!(123)
rng = Random.default_rng()

# Define a zero-mean banana-shaped distribution
struct Banana <: ContinuousMultivariateDistribution
    d::Int            # Dimension
    b::Float64        # Curvature
    Z::Float64        # Normalizing constant
    C::Matrix{Float64} # Covariance matrix
    C⁻¹::Matrix{Float64} # Inverse of covariance matrix
end

# Constructor with additional scaling parameter s
Banana(d::Int, b::Float64, s::Float64=100.0) = Banana(
    d, b, sqrt(s * (2π)^d), 
    Matrix(Diagonal(vcat(s, ones(d-1)))), 
    Matrix(Diagonal(vcat(1/s, ones(d-1))))
)

Base.length(d::Banana) = d.d

Base.eltype(::Banana) = Float64

Distributions.sampler(d::Banana) = d

# Define the transformation function φ and the inverse ϕ⁻¹ for the banana distribution
φ(x, b, s) = [x[1], x[2] + b * x[1]^2 - s * b]
ϕ⁻¹(y, b, s) = [y[1], y[2] - b * y[1]^2 + s * b]

function Distributions._rand!(rng::AbstractRNG, d::Banana, x::AbstractArray{<:Real})
    b, C = d.b, d.C
    mvnormal = MvNormal(zeros(2), C)
    for i in axes(x, 2)
        x[:, i] = φ(rand(rng, mvnormal), b, C[1, 1])
    end
    return x
end

function Distributions._logpdf(d::Banana, x::AbstractArray)
    Z, C⁻¹, b = d.Z, d.C⁻¹, d.b
    ϕ⁻¹_x = ϕ⁻¹(x, b, d.C[1, 1])
    return -log(Z) - 0.5 * ϕ⁻¹_x' * C⁻¹ * ϕ⁻¹_x
end

Distributions.mean(d::Banana) = zeros(d.d)
Distributions.var(d::Banana) = diag(d.C)
Distributions.cov(d::Banana) = d.C

function visualize(d::Banana, samples=rand(d, 1000))
    scatter(samples[1, :], samples[2, :], label="Samples", alpha=0.5, legend=:bottomright)
    xrange = range(minimum(samples[1, :])-1, maximum(samples[1, :])+1, length=100)
    yrange = range(minimum(samples[2, :])-1, maximum(samples[2, :])+1, length=100)
    z = [exp(Distributions.logpdf(d, [x, y])) for x in xrange, y in yrange]
    contour!(xrange, yrange, z', levels=15, color=:viridis, label="PDF", linewidth=2)
    return current()
end

function create_planar_flow(n_trans::Int, q₀)
    d = length(q₀)
    Ts = ∘([PlanarLayer(d) for _ in 1:n_trans]...)
    return transformed(q₀, Ts)
end

function elbo_single_sample(x, flow, logp, logq)
    y, logabsdetjac = with_logabsdet_jacobian(flow.transform, x)
    return logp(y) - logq(x) + logabsdetjac
end

function elbo(xs, flow::Bijectors.MultivariateTransformed, logp, logq)
    n_samples = size(xs, 2)
    elbo_values = map(x -> elbo_single_sample(x, flow, logp, logq), eachcol(xs))
    
    # log-sum-exp trick
    max_elbo = maximum(elbo_values)
    avg_elbo = log(sum(exp.(elbo_values .- max_elbo))) + max_elbo - log(n_samples)
    
    return avg_elbo
end

elbo(rng::AbstractRNG, flow::Bijectors.MultivariateTransformed, logp, logq, n_samples) = elbo(
    rand(rng, flow, n_samples), flow, logp, logq
)

function train_flatten!(
    rng::AbstractRNG, 
    flow::Bijectors.MultivariateTransformed, 
    p::Banana, 
    n_epochs::Int, 
    n_samples::Int
)
    logp = Base.Fix1(logpdf, p)
    logq = Base.Fix1(logpdf, flow.dist)

    flat, re = destructure(flow)
    rule = Optimisers.ADAM()
    st = Optimisers.setup(rule, flat)
    loss(x) = -elbo(rng, re(x), logp, logq, n_samples)
    losses = zeros(n_epochs)
    @showprogress 1 for i in 1:n_epochs
        losses[i] = loss(flat)
        ∇flat = gradient(loss, flat)
        st, flat = Optimisers.update!(st, flat, ∇flat[1])
    end
    return losses, re(flat)
end

function train!(
    rng::AbstractRNG,
    flow::Bijectors.MultivariateTransformed,
    p::Banana,
    n_epochs::Int,
    n_samples::Int
)
    logp = Base.Fix1(logpdf, p)
    logq = Base.Fix1(logpdf, flow.dist)

    rule = Optimisers.ADAM()
    st = Optimisers.setup(rule, flow)
    loss(flow) = -elbo(rng, flow, logp, logq, n_samples)
    losses = zeros(n_epochs)
    @showprogress 1 for i in 1:n_epochs
        losses[i] = loss(flow)
        ∇flow = gradient(loss, flow)[1]
        st, flow = Optimisers.update!(st, flow, ∇flow)
    end
    return losses, flow
end

##
banana_dist = Banana(2, 0.1)
flow = create_planar_flow(20, MvNormal(zeros(Float64, 2), I))
flow_untrained = deepcopy(flow)

losses, flow_trained = train!(Random.GLOBAL_RNG, flow, banana_dist, 30000, 100)
losses, flow_trained = train_flatten!(Random.GLOBAL_RNG, flow, banana_dist, 30000, 10)

##

function compare_trained_and_untrained_flow(flow_trained, flow_untrained, true_dist, n_samples)
    samples_trained = rand(flow_trained, n_samples)
    samples_untrained = rand(flow_untrained, n_samples)
    samples_true = rand(true_dist, n_samples)
    
    scatter(samples_true[1, :], samples_true[2, :], label="True Distribution", color=:blue, markersize=2, alpha=0.5)
    scatter!(samples_untrained[1, :], samples_untrained[2, :], label="Untrained Flow", color=:red, markersize=2, alpha=0.5)
    scatter!(samples_trained[1, :], samples_trained[2, :], label="Trained Flow", color=:green, markersize=2, alpha=0.5)
    
    xlabel!("X")
    ylabel!("Y")
    title!("Comparison of Trained and Untrained Flow")
    
    return current()
end

plot(losses, label="Loss", linewidth=2)
compare_trained_and_untrained_flow(flow_trained, flow_untrained, banana_dist, 1000)
