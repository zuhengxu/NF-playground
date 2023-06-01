using Distributions, LinearAlgebra, Plots
using Bijectors
using Bijectors: with_logabsdet_jacobian
using Flux, Zygote
using Optimisers
using ProgressMeter
using Random
Random.seed!(123)
rng = Random.default_rng()

include("AffineCoupling.jl")
# the banana distribution, we will be testing on 4d banana
include("../../logpdfs/Banana.jl")

function create_AffineCoupling_flow(Ls::Union{Vector{AffineCoupling}, AffineCoupling}, q₀)
    Ts = ∘(Ls...)
    return transformed(q₀, Ts)
end

function elbo_single_sample(x, flow::Bijectors.MultivariateTransformed, logp, logq)
    y, logabsdetjac = Bijectors.with_logabsdet_jacobian(flow.transform, x)
    return logp(y) - logq(x) + logabsdetjac
end

function elbo(xs, flow::Bijectors.MultivariateTransformed, logp, logq)
    n_samples = size(xs, 2)
    elbo_values = map(x -> elbo_single_sample(x, flow, logp, logq), eachcol(xs))
    return sum(elbo_values) / n_samples
end

elbo(rng::AbstractRNG, flow::Bijectors.MultivariateTransformed, logp, logq, n_samples) = elbo(
    rand(rng, flow.dist, n_samples), flow, logp, logq
)

################
# training
###############
function train!(
    rng::AbstractRNG,
    flow::Bijectors.MultivariateTransformed,
    p::Banana,
    n_epochs::Int,
    n_samples::Int
)
    logp(x) = logpdf(p, @views(x[1:2])) + logpdf(p, @views(x[3:4]))
    logq = Base.Fix1(logpdf, flow.dist)

    rule = Optimisers.ADAM()
    st = Optimisers.setup(rule, flow)
    loss(flow) = -elbo(rng, flow, logp, logq, n_samples)
    losses = zeros(n_epochs)
    @showprogress 1 for i in 1:n_epochs
        losses[i] = loss(flow)
        ∇flow = only(gradient(loss, flow))
        st, flow = Optimisers.update!(st, flow, ∇flow)
    end
    return losses, flow
end

##
banana_dist = Banana(2, 0.1)
visualize(banana_dist)
Ls = [
    AffineCoupling(4, 8, 1:2),
    AffineCoupling(4, 8, 3:4),
    AffineCoupling(4, 8, 1:2),
    AffineCoupling(4, 8, 3:4),
    # AffineCoupling(4, 8, 1:2),
    # AffineCoupling(4, 8, 3:4),
    # AffineCoupling(4, 8, 1:2),
    # AffineCoupling(4, 8, 3:4),
]


flow = create_AffineCoupling_flow(Ls, MvNormal(zeros(Float64, 4), I))
flow_untrained = deepcopy(flow)

losses, flow_trained = train!(Random.GLOBAL_RNG, flow, banana_dist, 100000, 1)


function compare_trained_and_untrained_flow(flow_trained, flow_untrained, true_dist, n_samples)
    samples_trained = rand(flow_trained, n_samples)
    samples_untrained = rand(flow_untrained, n_samples)
    samples_true = rand(true_dist, n_samples)
    
    scatter(samples_true[1, :], samples_true[2, :], label="True Distribution", color=:blue, markersize=2, alpha=0.5)
    scatter!(samples_untrained[1, :], samples_untrained[2, :], label="Untrained Flow", color=:red, markersize=2, alpha=0.5)
    scatter!(samples_trained[1, :], samples_trained[2, :], label="Trained Flow", color=:green, markersize=2, alpha=0.5)
    
    xlabel!("X")
    ylabel!("Y")
    title!("Comparison of Trained and Untrained Flow (first two dims)")
    
    return current()
end

plot(losses, label="Loss", linewidth=2)
compare_trained_and_untrained_flow(flow_trained, flow_untrained, banana_dist, 1000)

# Bijectors.with_logabsdet_jacobian(flow.transform, rand(4))