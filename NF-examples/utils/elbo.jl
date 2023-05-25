
"""
elbo estimation for realnvp using a single sample
"""
function single_elbo(flow::Bijectors.MultivariateTransformed, logp, logq)
    x = rand(flow.dist)
    y, logjac = with_logabsdet_jacobian(flow.transform, x) 
    el = logp(y) -logq(x) + logjac
    # x, y, logjac, logpdf_y = Bijectors.forward(flow) # this is deprecated
    # el = logp(y) - logpdf_y
    return el
end

"""
elbo estimation for realnvp using multiple samples
"""
function nf_ELBO(flow::Bijectors.MultivariateTransformed, logp, logq; elbo_size = 1)
    el = 0.0    
    for i in 1:elbo_size
        el += 1/elbo_size*single_elbo(flow, logp, logq)
    end
    return el
end

function nf(flow::Bijectors.MultivariateTransformed, logp, logq, niters::Int; elbo_size::Int = 1, optimizer = Flux.ADAM(1e-3), kwargs...)

    ps = Flux.params(flow)

    #define loss
    loss = () -> begin 
        elbo = nf_ELBO(flow, logp, logq; elbo_size = elbo_size)
        # elbo = single_elbo(flow, logp, logq)
        return -elbo
    end

    loss_log, _ = vi_train!(niters, loss, ps, optimizer; logging_ps = false, kwargs...)
    
    return [[copy(p) for p in ps]], -loss_log
end
