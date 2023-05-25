using Random
Random.seed!(123)
rng = Random.default_rng()

# using Gaussian mixture as general targets

# cross and 25 Gaussian

# Define a zero-mean banana-shaped distribution
struct Cross <: ContinuousMultivariateDistribution
    d::Int            # Dimension
    M::Matrix{Float64} # means of guassian mixture
    Cov::Matrix{Float64} # covariance matrix
end


struct twentyfiveGaussian <: ContinuousMultivariateDistribution
    d::Int            # Dimension
    M::Matrix{Float64} # means of guassian mixture
    Cov::Matrix{Float64} # covariance matrix
end 