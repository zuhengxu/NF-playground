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
    Matrix(Diagonal(vcat(s, ones(d - 1)))),
    Matrix(Diagonal(vcat(1 / s, ones(d - 1))))
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

# fixing this
function ∇logpdf(d::Banana, x::AbstractArray)
    b = d.b
    -[1 / 100 * x[1] + (x[2] - b * x[1]^2 + 100 * b) * (-2 * b * x[1]), x[2] - b * x[1]^2 + 100 * b]
end


Distributions.mean(d::Banana) = zeros(d.d)
Distributions.var(d::Banana) = diag(d.C)
Distributions.cov(d::Banana) = d.C

function visualize(d::Banana, samples=rand(d, 1000))
    xrange = range(minimum(samples[1, :]) - 1, maximum(samples[1, :]) + 1, length=100)
    yrange = range(minimum(samples[2, :]) - 1, maximum(samples[2, :]) + 1, length=100)
    z = [exp(Distributions.logpdf(d, [x, y])) for x in xrange, y in yrange]
    contour(xrange, yrange, z', levels=15, color=:viridis, label="PDF", linewidth=2)
    scatter!(samples[1, :], samples[2, :], label="Samples", alpha=0.3, legend=:bottomright)
    return current()
end
