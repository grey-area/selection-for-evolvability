include("fitnessfunctions.jl")

using PyPlot

function do_trial(trial::Int64, generations::Int64, population_size::Int64, fitness_function::LipsonFitnessFunction, results::Array{Float64, 2})
    task = 1

    networks = LipsonIndividual[LipsonIndividual() for i in 1:population_size]

    for generation in 1:generations

        delta_fitness_function!(fitness_function)

        fitnesses = Float64[evaluate_fitness(network, fitness_function) for network in networks]
        results[trial, generation] = mean(fitnesses)
        #println("$(mean(fitnesses)), $(maximum(fitnesses))")

        new_networks = LipsonIndividual[]
        sizehint!(new_networks, population_size)
        for p in 1:population_size
            i1 = rand(1:population_size)
            i2 = rand(1:population_size)
            parent_index = i1
            if fitnesses[i2] > fitnesses[i1]
                parent_index = i2
            end
            push!(new_networks,  mutated_copy(networks[parent_index]))
        end
        networks = new_networks
    end

end

function main()
    fitness_function = LipsonFitnessFunction()

    trials = 1
    generations = 50
    results = zeros(trials, generations)
    population_size = 1000

    for trial in 1:trials
        print("$trial      \r")
        do_trial(trial, generations, population_size, fitness_function, results)
    end
    println("")

    PyPlot.plot(mean(results, 1)')
    PyPlot.show()

end

main()


