type Particle
    num_particles::Int64
    x1s::Array{Float64,1}
    x2s::Array{Float64,1}
    weights::Array{Float64,1}
end

function init_particle(num_particles::Int64 = 1000)
    x1s = 10.0 + 10.0 * randn(num_particles)
    x2s = 10.0 + 10.0 * randn(num_particles)
    weights = log(ones(num_particles) / Float64(num_particles))
    particle = Particle(num_particles, x1s, x2s, weights)

    return particle
end

function particle_pred(particle::Particle)
    q = 1.0
    particle.x1s += randn(particle.num_particles) * √q
    particle.x2s += randn(particle.num_particles) * √q
    particle.x1s[particle.x1s .< 0] = 0.0001
    particle.x2s[particle.x2s .< 0] = 0.0001
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
function particle_update(particle::Particle, obs::Array{Float64,1}, sample_size::Int64, evolvability_type::ASCIIString)
    if evolvability_type == "variance"
        k = 0.5 * (sample_size - 1.0)
        theta1 = 2.0 * particle.x1s / Float64(sample_size - 1)
        theta1[theta1 .< 0] = 0.0000001
        theta2 = 2.0 * particle.x2s / Float64(sample_size - 1)
        theta2[theta2 .< 0] = 0.0000001
        particle.weights += -obs[1] ./theta1 + (-k) * log(theta1)
        particle.weights += -obs[2] ./theta2 + (-k) * log(theta2)
        particle.weights = normalize_weights(particle.weights)
    elseif evolvability_type == "std"
        particle.weights += -(obs[1] ./ particle.x1s).^2 * (1/pi + (sample_size-2)/2) - (sample_size+1) * log(particle.x1s)
        particle.weights += -(obs[2] ./ particle.x2s).^2 * (1/pi + (sample_size-2)/2) - (sample_size+1) * log(particle.x2s)
        particle.weights = normalize_weights(particle.weights)
    else
        sigma1 = (0.125 + 1.29 * (sample_size-1)^(-0.73)) * particle.x1s
        sigma2 = (0.125 + 1.29 * (sample_size-1)^(-0.73)) * particle.x2s

        particle.weights += -(obs[1] - particle.x1s).^2 ./ (2 * sigma1.^2) - log(particle.x1s)
        particle.weights += -(obs[2] - particle.x2s).^2 ./ (2 * sigma2.^2) - log(particle.x2s)

        particle.weights = normalize_weights(particle.weights)
    end
end

function particle_resample(particle::Particle)

    #s_eff = 1.0 / sum(particle.weights .^ 2)
    # test if s_eff < 0.5 * num particles
    # below is equivalent for log probabilities
    if log_sum_exp_log_probs(2*particle.weights) > log(2) - log(particle.num_particles)
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
        particle.x1s = getindex(particle.x1s, indices)
        particle.x2s = getindex(particle.x2s, indices)
        particle.weights = log(ones(particle.num_particles) / particle.num_particles)
    end
end

function particle_duplicate(particle::Particle, index::Int64)
    if index==1
        particle.x2s = copy(particle.x1s)
    else
        particle.x1s = copy(particle.x2s)
    end
end
