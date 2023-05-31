using Flux
using Functors
using Bijectors
using Bijectors:∘, PartitionMask


struct Coupling{F, M} <: Bijector where {F, M <: PartitionMask}
    θ::F
    mask::M
end
@functor Coupling (θ,)

function Coupling(θ, n::Int)
    idx = div(n, 2)
    return Coupling(θ, PartitionMask(n, 1:idx))
end

function Coupling(cl::Coupling, mask::PartitionMask)
    return Coupling(cl.θ, mask)
end

cl1 = Coupling(Dense(10,10), 5)
cl2 = Coupling(Dense(10,10), 5)

cl = cl1 ∘ cl2
which(∘, (typeof(cl1), typeof(cl2)))