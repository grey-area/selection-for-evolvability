type Particle
    num_particles::Int64
    K::Int64
    xs::Array{Float64,2}
    qs::Array{Float64, 1}
    weights::Array{Float64,1}
end

function init_particle(num_particles::Int64 = 1000, K::Int64 = 2)
    xs = 10.0 + sqrt(100.0) * randn(K, num_particles)
    qs = 5.0 * randexp(num_particles)
    weights = log(ones(num_particles) / Float64(num_particles))
    particle = Particle(num_particles, K, xs, qs, weights)
    return particle
end

function filter_predict(particle::Particle, ML_q::Float64, q_inference_type::Int64, evolvability_type::AbstractString, population_index::Int64)
    q = 1.0
    if q_inference_type == 1
        q = ML_q
    end

    if q_inference_type != 2
        particle.xs[population_index, :] += randn(1, particle.num_particles) * √q
    else
        particle.xs[population_index, :] += randn(1, particle.num_particles) .* √particle.qs'
        particle.qs += randn(particle.num_particles) * 0.05
        particle.qs[particle.qs .< 0] = 0.0001
    end

    #if evolvability_type != "maximum"
    # TODO otherwise the likelihood term for max below tries to take the log of negative numbers. Investigate
    particle.xs[particle.xs .< 0] = 0.0001
    #end
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

# TODO sample_size should be array
# Given a pair of observations, add log likelihoods (up to additive constant) to the particle weights
function filter_update(particle::Particle, obs1::Array{Float64,1}, sample_sizes::Array{Int64,1}, evolvability_type::AbstractString, population_index::Int64)

    for (obs, sample_size) in zip(obs1, sample_sizes)
        if evolvability_type == "variance"
            k = 0.5 * (sample_size - 1.0)
            thetas = 2.0 * particle.xs[population_index, :] / Float64(sample_size - 1)
            thetas[thetas .< 0] = 0.0000001
            particle.weights += squeeze(-obs ./ thetas +(-k) * log(thetas), 1)
        elseif evolvability_type == "std"
            #particle.weights += squeeze(-(obs ./ particle.xs[population_index, :]).^2 * (1/pi + (sample_size-2)/2) - (sample_size+1) * log(particle.xs[population_index, :]), 1)
            particle.weights += squeeze(-(sample_size - 2 + 1/(pi-2)) * (obs ./ particle.xs[population_index, :] .- 1).^2 - log(particle.xs[population_index, :]), 1)
        else
            sigmas = (0.125 + 1.29 * (sample_size-1)^(-0.73)) * particle.xs[population_index, :]
            particle.weights += squeeze(- (obs - particle.xs[population_index, :]) .^ 2 ./ (2 * sigmas .^ 2) - log(sigmas), 1)
            #particle.weights += squeeze(-(obs - particle.xs[population_index, :]) .^ 2 ./ (2 * sigmas .^ 2) - log(particle.xs[population_index, :]), 1)
        end

        particle.weights = normalize_weights(particle.weights)
    end

    particle_resample(particle)
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

function filter_expected_values(particle::Particle)
    return sum(particle.xs .* exp(particle.weights'), 2)
end

function filter_probabilities(particle::Particle)
    maxes = mod(squeeze(findmax(particle.xs, 1)[2], 1) - 1, particle.K) + 1
    probs = zeros(particle.K)
    exp_weights = exp(particle.weights)
    for (max, exp_weight) in zip(maxes, exp_weights)
        probs[max] += exp_weight
    end
    probs /= sum(probs)
end

function filter_duplicate(particle::Particle, index::Int64)
    particle.xs = repmat(particle.xs[index, :], particle.K)
end
