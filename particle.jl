type Particle
    num_particles::Int64
    x1s::Array{Float64,1}
    x2s::Array{Float64,1}
    weights::Array{Float64,1}
end

function init_particle(num_particles::Int64 = 1000)
    x1s = randn(num_particles) * √10.0 + 2.0
    x2s = randn(num_particles) * √10.0 + 2.0
    weights = ones(num_particles) / Float64(num_particles)
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


function particle_update(particle::Particle, obs::Array{Float64,1}, sample_size::Int64, evolvability_type::ASCIIString)

    if evolvability_type == "variance"
        k = 0.5 * (sample_size - 1.0)
        theta1 = 2.0 * particle.x1s / Float64(sample_size - 1)
        theta2 = 2.0 * particle.x2s / Float64(sample_size - 1)
        theta1[theta1 .< 0] = 0.0000001
        theta2[theta2 .< 0] = 0.0000001
        # TODO treat NaNs as zeros?
        particle.weights .*= exp(- obs[1] ./ theta1) .* obs[1] ^ (k-1) .* theta1 .^ (-k)
        particle.weights /= sum(particle.weights)
        particle.weights .*= exp(- obs[2] ./ theta2) .* obs[2] ^ (k-1) .* theta2 .^ (-k)
        #particle.weights .*= exp(-(obs[1] * obs[2]) ./ (theta1 .* theta2)) .* (obs[1] * obs[2])^(k-1) .* (theta1 .* theta2).^(-k)
        particle.weights /= sum(particle.weights)
    else
        sigma = (0.125 + 1.29 * (sample_size-1)^(-0.73)) .* particle.x1s
        particle.weights .*= (1./(√(2pi) * sigma)) .* exp(-0.5 * ((particle.x1s-obs[1]) ./ sigma).^2 )
        sigma = (0.125 + 1.29 * (sample_size-1)^(-0.73)) .* particle.x2s
        particle.weights /= sum(particle.weights)
        particle.weights .*= (1./(√(2pi) * sigma)) .* exp(-0.5 * ((particle.x2s-obs[2]) ./ sigma).^2 )
        particle.weights /= sum(particle.weights)
    end
end

function particle_resample(particle::Particle)
    s_eff = 1.0 / sum(particle.weights .^ 2)
    if s_eff < 0.5 * particle.num_particles
        positions = (collect(0:(particle.num_particles-1)) + rand()) / particle.num_particles
        indices = zeros(Int64, particle.num_particles)
        cumulative_sums = cumsum(particle.weights)
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
        particle.weights = ones(particle.num_particles) / particle.num_particles
    end
end

function particle_duplicate(particle::Particle, index::Int64)
    if index==1
        particle.x2s = copy(particle.x1s)
    else
        particle.x1s = copy(particle.x2s)
    end
end
