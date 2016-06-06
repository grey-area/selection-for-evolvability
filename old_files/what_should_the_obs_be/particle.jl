using Distributions

type Particle
    num_particles::Int64
    x1s::Array{Float64,1}
    weights::Array{Float64,1}
end

function init_particle(num_particles::Int64 = 1000)
    x1s = rand(Gamma(1, 2), num_particles)
    weights = log(ones(num_particles) / Float64(num_particles))
    particle = Particle(num_particles, x1s, weights)

    return particle
end

#=
function particle_pred(particle::Particle)
    q = 1.0
    particle.x1s += randn(particle.num_particles) * √q
    particle.x2s += randn(particle.num_particles) * √q
    particle.x1s[particle.x1s .< 0] = 0.0001
    particle.x2s[particle.x2s .< 0] = 0.0001
end
=#

# Calculate the log of the sum of probabilities given the log probabilities, while avoiding overflow
function log_sum_exp_log_probs(log_probs::Array{Float64, 1})
    max_log_prob = maximum(log_probs)
    return max_log_prob + log(sum(exp(log_probs - max_log_prob)))
end

# Normalize a list of log probabilities so the probabilities sum to 1
function normalize_weights(weights::Array{Float64,1})
    return weights - log_sum_exp_log_probs(weights)
end

# Update when we observe the standard deviation of the offspring
function particle_update1(particle::Particle, obs::Float64, sample_size::Int64)
    particle.weights += -(obs ./ particle.x1s).^2 * (1/pi + (sample_size-2)/2) - (sample_size+1) * log(particle.x1s)
    particle.weights = normalize_weights(particle.weights)
end

# Update when we observe just a single sample
function particle_update2(particle::Particle, obs::Float64)
    particle.weights += - 0.5 * (obs ./ particle.x1s).^2 - log(particle.x1s)
    particle.weights = normalize_weights(particle.weights)
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
        particle.weights = log(ones(particle.num_particles) / particle.num_particles)
    end
end

trials = 500
times = 100
sample_size = 2
errors1 = zeros(times, trials)
errors2 = zeros(times, trials)

samps = zeros(times, trials)

for trial in 1:trials
    println(trial)
    particle1 = init_particle()
    particle2 = init_particle()

    for time in 1:times
        samples = randn(sample_size)
        particle_update1(particle1, std(samples), sample_size)
        particle_resample(particle1)
        particle_update1(particle2, sqrt(0.5*pi) * std(samples), sample_size)
        particle_resample(particle2)

        #=
        for sample in samples
            particle_update2(particle2, sample)
            particle_resample(particle2)
        end
        =#

        #=
        for sample_i in 1:div(length(samples), 2)
            mini_sample = sqrt(0.5*pi) * std(samples[(sample_i-1)*2+1:(sample_i-1)*2+2])
            samps[time, trial] = mini_sample
            particle_update1(particle2, mini_sample, 2)
            particle_resample(particle2)
        end
        =#

        pred1 = sum(particle1.x1s .* exp(particle1.weights))
        pred2 = sum(particle2.x1s .* exp(particle2.weights))
        error1 = abs(pred1 - 1.0)
        error2 = abs(pred2 - 1.0)
        errors1[time, trial] = error1
        errors2[time, trial] = error2
    end

end

println(mean(samps))

using PyPlot

PyPlot.plot(mean(errors1, 2))
PyPlot.plot(mean(errors2, 2))
PyPlot.show()
