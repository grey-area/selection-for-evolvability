type Particle
    num_particles::Int64
    K::Int64
    xs::Array{Float64,2}
    qs::Array{Float64, 1}
    weights::Array{Float64,1}
end

function init_particle(num_particles::Int64 = 1000, K::Int64 = 2)
    xs = 10.0 + 10.0 * randn(K, num_particles)
    qs = 5.0 * randexp(num_particles)
    weights = log(ones(num_particles) / Float64(num_particles))
    particle = Particle(num_particles, K, xs, qs, weights)
    return particle
end

# TODO use the thompson_i info
function particle_pred(particle::Particle, ML_q::Float64, q_inference_type::Int64, thompson_i::Int64)
    q = 1.0
    if q_inference_type == 1
        q = ML_q
    end

    if q_inference_type != 2
        particle.xs += randn(particle.K, particle.num_particles) * √q
    else
        particle.xs += randn(particle.K, particle.num_particles) .* √particle.qs'
        particle.qs += randn(particle.num_particles) * 0.05
        particle.qs[particle.qs .< 0] = 0.0001
    end

    particle.xs[particle.xs .< 0] = 0.0001
end

# Calculate the log of the sum of probabilities given the log probabilities, while avoiding overflow
function log_sum_exp_log_probs(log_probs::Array{Float64, 1})
    max_log_prob = maximum(log_probs)
    return max_log_prob + log(sum(exp(log_probs - max_log_prob)))
end

# Normalize a list of log probabilities so the probabilities sum to 1
function normalize_weights(weights::Array{Float64,1})
    return weights - log_sum_exp_log_probs(weights)
end

# Given a pair of observations, add log likelihoods (up to additive constant) to the particle weights
function particle_update(particle::Particle, obs1::Array{Float64,2}, sample_size::Int64, evolvability_type::ASCIIString, thompson_i::Int64)

    for row in 1:size(obs1)[2]
        obs = obs1[:, row]

        if evolvability_type == "variance"
            k = 0.5 * (sample_size - 1.0)
            thetas = 2.0 * particle.xs / Float64(sample_size - 1)
            thetas[thetas .< 0] = 0.0000001

            if thompson_i == 0
                particle.weights += squeeze(sum(-obs ./ thetas +(-k) * log(thetas), 1), 1)
            else
                particle.weights += squeeze(-obs[thompson_i] ./ thetas[thompson_i, :] +(-k) * log(thetas[thompson_i, :]), 1)
            end

            particle.weights = normalize_weights(particle.weights)

        elseif evolvability_type == "std"
            if thompson_i == 0
                particle.weights += squeeze(sum(-(obs ./ particle.xs).^2 * (1/pi + (sample_size-2)/2) - (sample_size+1) * log(particle.xs), 1), 1)
            else
                particle.weights += squeeze(-(obs[thompson_i] ./ particle.xs[thompson_i, :]).^2 * (1/pi + (sample_size-2)/2) - (sample_size+1) * log(particle.xs[thompson_i, :]), 1)
            end
            particle.weights = normalize_weights(particle.weights)

        else
            sigmas = (0.125 + 1.29 * (sample_size-1)^(-0.73)) * particle.xs

            if thompson_i == 0
                particle.weights += squeeze(sum(-(obs .- particle.xs) .^ 2 ./ (2 * sigmas .^ 2) - log(particle.xs), 1), 1)
            else
                particle.weights += squeeze(-(obs[thompson_i] - particle.xs[thompson_i, :]) .^ 2 ./ (2 * sigmas[thompson_i, :] .^ 2) - log(particle.xs[thompson_i, :]), 1)
            end

            particle.weights = normalize_weights(particle.weights)
        end

    end
end

function particle_resample(particle::Particle)

    #s_eff = 1.0 / sum(particle.weights .^ 2)
    # test if s_eff < 0.5 * (num particles + 1)
    # below is equivalent for log probabilities
    if log_sum_exp_log_probs(2*particle.weights) > log(2) - log(particle.num_particles + 1)
        #println("Resampling!")
        positions = (collect(0:(particle.num_particles-1)) + rand()) / particle.num_particles
        indices = zeros(Int64, particle.num_particles)
        cumulative_sums = cumsum(exp(particle.weights))
        cumulative_sums[end]  = 1.0
        i, j = 1, 1
        while i <= particle.num_particles
            if positions[i] < cumulative_sums[j]
                indices[i] = j
                i += 1
            else
                j += 1
            end
        end
        particle.xs = particle.xs[:, indices]
        particle.qs = particle.qs[indices]
        particle.weights = log(ones(particle.num_particles) / particle.num_particles)
    end
end

function particle_duplicate(particle::Particle, index::Int64)
    particle.xs = repmat(particle.xs[index, :], particle.K)
end
