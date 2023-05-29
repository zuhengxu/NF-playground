using Flux
using Functors
using Bijectors

struct RealNVPLayer <: Bijectors.Bijector
    D::Int
    # Mask::Bijectors.PartitionMask
    s::Chain
    t::Chain
end

@functor RealNVPLayer (s, t)

function RealNVPLayer(D::Int, dims::Int)
    s = Chain(Dense(D, dims, relu), Dense(dims, D))
    t = Chain(Dense(D, dims, relu), Dense(dims, D))
    RealNVPLayer(D, s, t)
end

r = RealNVPLayer(10, 20)
p = Flux.params(r)

r2 = RealNVPLayer(10, 20)

T = ∘(r, r2)

t1 = Bijectors.∘([PlanarLayer(20) for _ in 1:2]...)
t2 = Bijectors.∘([PlanarLayer(20) for _ in 1:2]...)

import Bijectors
T = Bijector.∘(t1, t2)
T1 = ∘(r, r2)
T1(randn(20))


Bijectors.forward(T, randn(20))

function test(x)
    sum(abs2, T1(x))
end

using Zygote 
gradient(test, rand(20))

function (r::RealNVPLayer)(x::AbstractArray)
    d = size(x, 1) ÷ 2
    x₁, x₂ = x[1:d, :], x[d+1:end, :]
    y₁ = x₁
    y₂ = x₂ .* exp.(r.s(x₁)) .+ r.t(x₁)
    vcat(y₁, y₂)
end

# function Bijectors.forward(r::RealNVPLayer, )


function (r_inverse::RealNVPLayer)(y::AbstractArray)
    d = size(y, 1) ÷ 2
    y₁, y₂ = y[1:d, :], y[d+1:end, :]
    x₁ = y₁
    x₂ = (y₂ .- r.t(y₁)) ./ exp.(r.s(y₁))
    vcat(x₁, x₂)
end