using Random
Random.seed!(123)
rng = Random.default_rng()

# Define a zero-mean banana-shaped distribution
struct NealsFunnel <: ContinuousMultivariateDistribution
    d::Int            # Dimension
    b::Float64        # Curvature
    Z::Float64        # Normalizing constant
    C::Matrix{Float64} # Covariance matrix 
end