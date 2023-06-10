using Flux
using Functors
using Bijectors
using Bijectors: ∘, partition, PartitionMask
import Bijectors: transform, with_logabsdet_jacobian, logabsdetjac, rqs_univariate, rqs_univariate_inverse, rqs_logabsdetjac


"""
Neural Rational quadratic Spline layer "(https://proceedings.neurips.cc/paper_files/paper/2019/file/7ac71d433f282034e088473244df8c02-Paper.pdf)"
"""
struct NeuralSplineLayer{T} <: Bijectors.Bijector
    D::Int
    Mask::Bijectors.PartitionMask
    w::T # width (xs)
    h::T # height (ys)
    d::T # derivative of the knots
    B::Real # bound of the knots
end

function MLP_3layer(input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
    Chain(Flux.Dense(input_dim, hdims, activation), Flux.Dense(hdims, hdims, activation), Flux.Dense(hdims, output_dim))
end

function NeuralSplineLayer(
    D::Int,  # dimension of input
    hdims::Int, # dimension of hidden units for s and t
    K::Int, # number of knots
    mask_idx::AbstractVector, # index of dimensione that one wants to apply transformations on
    B::Real # bound of the knots
)
    num_of_transformed_dims = length(mask_idx)
    input_dims = D - num_of_transformed_dims
    w = [MLP_3layer(input_dims, hdims, K) for i in 1:num_of_transformed_dims]
    h = [MLP_3layer(input_dims, hdims, K) for i in 1:num_of_transformed_dims]
    d = [MLP_3layer(input_dims, hdims, K - 1) for i in 1:num_of_transformed_dims]
    Mask = Bijectors.PartitionMask(D, mask_idx)
    NeuralSplineLayer(D, Mask, w, h, d, B)
end

# let params track field (w, h, d)
@functor NeuralSplineLayer (w, h, d)

# define forward and inverse transformation
function instantiate_rqs(nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}}, x::AbstractVector)
    # instantiate rqs knots and derivatives
    # TODO: make pullreqeust: Bijectors.RationalQuadraticSpline doesn't allow input Float32 (not typestable line )
    ws = Float64.(reduce(hcat, [w(x) for w in nsl.w]))
    hs = Float64.(reduce(hcat, [h(x) for h in nsl.h]))
    ds = Float64.(reduce(hcat, [d(x) for d in nsl.d]))
    # TODO: need to ask whether there is a better way
    return Bijectors.RationalQuadraticSpline(ws', hs', ds', nsl.B)
end

function Bijectors.transform(nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}}, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.Mask, x)
    # TODO: need to ask whether there is a better way
    # instantiate rqs knots and derivatives
    Rqs = instantiate_rqs(nsl, x_2)
    y_1 = transform(Rqs, x_1)
    return Bijectors.combine(nsl.Mask, y_1, x_2, x_3)
end

function Bijectors.transform(insl::Inverse{<:NeuralSplineLayer{<:Vector{<:Flux.Chain}}}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.Mask, y)
    # todo: improve
    Rqs = instantiate_rqs(nsl, y2)
    x1 = transform(Inverse(Rqs), y1)
    return combine(nsl.Mask, x1, y2, y3)
end

(nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}})(x::AbstractVector) = Bijectors.transform(nsl, x)

# define logabsdetjac
function Bijectors.logabsdetjac(nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}}, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.Mask, x)
    # TODO: need to ask whether there is a better way
    # instantiate rqs knots and derivatives
    Rqs = instantiate_rqs(nsl, x_2)
    logjac = logabsdetjac(Rqs, x_1)
    return logjac
end

function Bijectors.logabsdetjac(insl::Inverse{<:NeuralSplineLayer{<:Vector{<:Flux.Chain}}}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.Mask, y)
    # todo: improve
    Rqs = instantiate_rqs(nsl, y2)
    logjac = logabsdetjac(Inverse(Rqs), y1)
    return logjac
end

function Bijectors.with_logabsdet_jacobian(nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}}, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.Mask, x)
    # TODO: need to ask whether there is a better way
    # instantiate rqs knots and derivatives
    Rqs = instantiate_rqs(nsl, x_2)
    y_1, logjac = with_logabsdet_jacobian(Rqs, x_1)
    return Bijectors.combine(nsl.Mask, y_1, x_2, x_3), logjac
end



# r1 = NeuralSplineLayer(4, 10, 10, 1:2, 2.0)
# r2 = NeuralSplineLayer(4, 10, 10, 3:4, 2.0) 
# T = r1 ∘ r2

# r1(randn(Float32, 4))
# Flux.params(T)
# T(randn(4))
# Bijectors.transform(r1, randn(4))


# K = 10; B = 2
# ws = randn(K)
# hs = randn(K)
# ds = randn(K-1)
# Ws = randn(3, K)
# Hs = randn(3, K)
# Ds = randn(3, K-1)
# ft = Float32
# rqs = Bijectors.RationalQuadraticSpline(ws, hs, ds, B)
# rqs1 = Bijectors.RationalQuadraticSpline(ft.(Ws), ft.(Hs), ft.(Ds), B)
