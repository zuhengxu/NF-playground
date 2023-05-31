using Flux
using Functors
using Bijectors
using Bijectors:∘

struct AffineCoupling <: Bijectors.Bijector
    D::Int
    Mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

# let params track field s and t
@functor AffineCoupling (s, t)

function AffineCoupling(D::Int, hdims::Int, mask_idx::AbstractVector)
    coupling_dim = D ÷ 2 # D is even
    s = Chain(Dense(coupling_dim, hdims, leakyrelu), Dense(hdims, coupling_dim))
    t = Chain(Dense(coupling_dim, hdims, leakyrelu), Dense(hdims, coupling_dim))
    Mask = Bijectors.PartitionMask(D, mask_idx)
    AffineCoupling(D, Mask, s, t)
end

function (af::AffineCoupling)(x::AbstractArray)
    # partition vector using 'af.mask::PartitionMask`
    x₁, x₂, x₃= Bijectors.partition(af.Mask, x)
    y₁ = x₁ .* af.s(x₂) .+ af.t(x₂)
    return Bijectors.combine(af.Mask, y₁, x₂, x₃)
end


function with_logabsdet_jacobian(af::AffinCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
    logjac = sum(log∘abs, af.s(x_1))
    return combine(af.mask, y_1, x_2, x_3), logjac
end

function with_logabsdet_jacobian(iaf::Inverse{<:AffineCoupling}, y::AbstractVector)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2) 
    logjac = -sum(log∘abs, af.s(x_1))
    return combine(af.mask, x_1, y_2, y_3), logjac
end


function logabsdetjac(cl::Coupling, x::AbstractVector)
    x_1, x_2, x_3 = partition(cl.mask, x)
    logjac = sum(log∘abs, af.s(x_1))
    return logjac
end

# D = 10
# r1 = AffineCoupling(D, 20, 1:5)
# r2 = AffineCoupling(D, 20, 6:10)

# T = r1∘r2
# T(randn(10))


# transform



# _logabsdetjac_scale 








