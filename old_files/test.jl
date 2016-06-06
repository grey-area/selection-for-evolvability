
type Particle1
    num_particles::Int64
    K::Int64
    xs::Array{Float64,2}
    qs::Array{Float64, 1}
    weights::Array{Float64,1}
end

function init_particle1(num_particles::Int64 = 1000, K::Int64 = 2)
    xs = 10.0 + 10.0 * randn(K, num_particles)
    qs = 5.0 * randexp(num_particles)
    weights = log(ones(num_particles) / Float64(num_particles))
    particle = Particle1(num_particles, K, xs, qs, weights)
    return particle
end

# TODO use the thompson_i info
function particle_pred1(particle::Particle1, ML_q::Float64, q_inference_type::Int64, thompson_i::Int64)
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



function particle_resample1(particle::Particle1)

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

function particle_duplicate1(particle::Particle1, index::Int64)
    particle.xs = repmat(particle.xs[index, :], particle.K)
end












































type Particle2
    num_particles::Int64
    x1s::Array{Float64,1}
    x2s::Array{Float64,1}
    qs::Array{Float64, 1}
    weights::Array{Float64,1}
end

function init_particle2(num_particles::Int64 = 1000)
    x1s = 10.0 + 10.0 * randn(num_particles)
    x2s = 10.0 + 10.0 * randn(num_particles)
    qs = 5.0 * randexp(num_particles)
    weights = log(ones(num_particles) / Float64(num_particles))
    particle = Particle2(num_particles, x1s, x2s, qs, weights)

    return particle
end

function particle_pred2(particle::Particle2, ML_q::Float64, q_inference_type::Int64, thompson_i::Int64)
    q = 1.0
    if q_inference_type == 1
        q = ML_q
    end

    if q_inference_type != 2
        particle.x1s += randn(particle.num_particles) * √q
        particle.x2s += randn(particle.num_particles) * √q
    else
        particle.x1s += randn(particle.num_particles) .* √particle.qs
        particle.x2s += randn(particle.num_particles) .* √particle.qs
        particle.qs += randn(particle.num_particles) * 0.05
        particle.qs[particle.qs .< 0] = 0.0001
    end

    particle.x1s[particle.x1s .< 0] = 0.0001
    particle.x2s[particle.x2s .< 0] = 0.0001
end




function particle_resample2(particle::Particle2)

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
        particle.qs = getindex(particle.qs, indices)
        particle.weights = log(ones(particle.num_particles) / particle.num_particles)
    end
end

function particle_duplicate2(particle::Particle2, index::Int64)
    if index==1
        particle.x2s = copy(particle.x1s)
    else
        particle.x1s = copy(particle.x2s)
    end
end









































































# Given a pair of observations, add log likelihoods (up to additive constant) to the particle weights
function particle_update1(particle::Particle1, obs1::Array{Float64,2}, sample_size::Int64, evolvability_type::ASCIIString, thompson_i::Int64)

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







# Given a pair of observations, add log likelihoods (up to additive constant) to the particle weights
function particle_update2(particle::Particle2, obs1::Array{Float64,2}, sample_size::Int64, evolvability_type::ASCIIString, thompson_i::Int64)

    for row in 1:size(obs1)[2]
        obs = obs1[:, row]

        if evolvability_type == "variance"
            k = 0.5 * (sample_size - 1.0)
            theta1 = 2.0 * particle.x1s / Float64(sample_size - 1)
            theta1[theta1 .< 0] = 0.0000001
            theta2 = 2.0 * particle.x2s / Float64(sample_size - 1)
            theta2[theta2 .< 0] = 0.0000001
            if thompson_i == 0 || thompson_i == 1
                particle.weights += -obs[1] ./theta1 + (-k) * log(theta1)
            end
            if thompson_i == 0 || thompson_i == 2
                particle.weights += -obs[2] ./theta2 + (-k) * log(theta2)
            end
            particle.weights = normalize_weights(particle.weights)
        elseif evolvability_type == "std"
            if thompson_i == 0 || thompson_i == 1
                particle.weights += -(obs[1] ./ particle.x1s).^2 * (1/pi + (sample_size-2)/2) - (sample_size+1) * log(particle.x1s)
            end
            if thompson_i == 0 || thompson_i == 2
                particle.weights += -(obs[2] ./ particle.x2s).^2 * (1/pi + (sample_size-2)/2) - (sample_size+1) * log(particle.x2s)
            end
            particle.weights = normalize_weights(particle.weights)
        else
            sigma1 = (0.125 + 1.29 * (sample_size-1)^(-0.73)) * particle.x1s
            sigma2 = (0.125 + 1.29 * (sample_size-1)^(-0.73)) * particle.x2s

            if thompson_i == 0 || thompson_i == 1
                particle.weights += -(obs[1] - particle.x1s).^2 ./ (2 * sigma1.^2) - log(particle.x1s)
            end
            if thompson_i == 0 || thompson_i == 2
                particle.weights += -(obs[2] - particle.x2s).^2 ./ (2 * sigma2.^2) - log(particle.x2s)
            end

            particle.weights = normalize_weights(particle.weights)
        end

    end
end
























particle1 = init_particle1(500, 2)
particle2 = init_particle2(500)

for i in 1:2
    particle_pred1(particle1, 1.0, 1, 0)
    particle_pred2(particle2, 1.0, 1, 0)
end

obs = [5., 5.]''

particle1.xs = [copy(particle2.x1s) copy(particle2.x2s)]'

for i in 1:1000
    particle_update1(particle1, obs, 10, "maximum", 0)
    particle_update2(particle2, obs, 10, "maximum", 0)

    #particle1.weights = copy(particle2.weights)

    particle_resample1(particle1)
    particle_resample2(particle2)
end

println("")

println( mean(particle1.xs[1,:]) )
println( mean(particle2.x1s) )
println("")

println( mean(particle1.xs[2,:]) )
println( mean(particle2.x2s) )
println("")

println( std(particle1.xs[1,:]) )
println( std(particle2.x1s) )
println("")

println( std(particle1.xs[2,:]) )
println( std(particle2.x2s) )

println("")

#=
for (i,j) in zip(exp(particle1.weights), exp(particle2.weights))
    print(i)
    print("\t")
    println(j)
end
=#

