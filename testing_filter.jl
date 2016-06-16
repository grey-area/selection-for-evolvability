include("kalman.jl")
include("kalman2.jl")
include("particle.jl")
include("particle2.jl")

# First, single variable
K = 3
evolvability_type = "std"

kalman1 = init_kalman(K)
kalman2 = init_kalman2(K)
particle1 = init_particle(5000, K)
particle2 = init_particle2(5000, K)

function print_filters()
    println("Kalman")
    for (name, kalman) in zip(["new implementation", "old implementation"], [kalman1, kalman2])
        println(name)
        for i in 1:K
            print(kalman.xs[i])
            print(", ")
            println(kalman.ps[i,i])
        end
    end

    println("Particle")
    for (name, particle) in zip(["new implementation", "old_implementation"], [particle1, particle2])
        println(name)
        for i in 1:K
            mean = squeeze(sum(particle.xs[i, :] .* exp(particle.weights'),2), 2)
            print(mean)
            print(", ")

            var = squeeze(sum(((particle.xs[i, :] .- mean) .^ 2) .* exp(particle.weights'), 2), 2)
            println(var)
        end
    end

    println("")
end

function predict()
    println("Predict\n")

    for population_index in 1:K
        filter_predict(kalman1, 0.0, 0, evolvability_type, population_index)
        filter_predict(particle1, 0.0, 0, evolvability_type, population_index)
    end

    kalman2_pred(kalman2, 0.0, 0)
    particle2_pred(particle2, 0.0, 0, 0)
end

function update()
    println("Update\n")

    K1 = K-2
    sigma = 110
    number_of_obs = 10
    sample_size = 2
    sample_sizes = ones(Int64, number_of_obs) * sample_size
    if evolvability_type == "std"
        observations = squeeze(std(sigma * randn(K1, number_of_obs, sample_size), 3), 3)
        correction_term = sqrt(2.0/(sample_size-1)) * gamma(sample_size/2) / gamma((sample_size-1)/2)
        observations = observations ./= correction_term
    elseif evolvability_type == "maximum"
        observations = squeeze(findmax(sigma * randn(K1, number_of_obs, sample_size), 3)[1], 3)
    else
        observations =squeeze(var(sqrt(sigma) * randn(K1, number_of_obs, sample_size), 3), 3)
    end

    #println(observations)

    for population_index in 1:K1
        obs = squeeze(observations[population_index, :], 1)
        filter_update(kalman1, obs, sample_sizes, evolvability_type, population_index)
        filter_update(particle1, obs, sample_sizes, evolvability_type, population_index)
    end

    #kalman2_update(kalman2, observations, sample_size, evolvability_type)
    #particle2_update(particle2, observations, sample_size, evolvability_type, 0)
    #particle2_resample(particle2)

end

for i in 1:1000
    print_filters()
    predict()
    print_filters()
    update()
    if mod(i, 10) == 0
        filter_duplicate(kalman1, 1)
        filter_duplicate(particle1, 1)
        println("\n\n\n\n\n\n\n\n\n\n")
    end
end
