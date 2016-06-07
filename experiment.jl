include("kalman.jl")
include("particle.jl")
include("fitnessfunctions.jl")

using StatsBase

function observe_evolvability(parent_fitness::Float64, offspring_fitnesses::Array{Float64, 1}, evolvability_type::AbstractString, sample_size::Int64)
	if evolvability_type == "variance"
		evolvability_observation = var(offspring_fitnesses)
	elseif evolvability_type == "std"
		correction_term = sqrt(2.0/(sample_size-1)) * gamma(sample_size/2) / gamma((sample_size-1)/2)
        evolvability_observation = std(offspring_fitnesses) / correction_term
    else
    	fitness_differences = offspring_fitnesses - parent_fitness
    	evolvability_observation = maximum(fitness_differences)
    end
    return evolvability_observation::Float64
end

function tournament_selection{T<:Union{SimpleIndividual, ReisingerIndividual, LipsonIndividual}}(N::Int64, N2::Int64, current_population_index::Int64, fitnesses::Array{Float64, 2}, populations::Array{T, 2}, fitness_function::FitnessFunction, evolvability_type::AbstractString)
    # array for new population, array of their fitnesses, array of arrays of fitnesses, array of sample sizes
    new_population = Array{T}(N2)
    new_fitnesses = zeros(N2)
    parent_offspring_fitnesses = [Array{Float64}(0) for i in 1:N]
    for i in 0:(div(N2, 2) - 1)
        i1 = rand(1:N)
        i2 = rand(1:N)
        fitter_index = i1
        if fitnesses[current_population_index, i2] > fitnesses[current_population_index, i1]
            fitter_index = i2
        end
        for j in 0:1
            new_index = 2i + j + 1
            new_population[new_index] = mutated_copy(populations[current_population_index, fitter_index])
            new_fitnesses[new_index] = evaluate_fitness(new_population[new_index], fitness_function)
            push!(parent_offspring_fitnesses[fitter_index], new_fitnesses[new_index])
        end
    end
    sample_sizes = Int64[length(fitnesses) for fitnesses in parent_offspring_fitnesses]

    # Now, for each non-zero entry in sample sizes, we need an evolvability observation
    evolvability_observations = Float64[observe_evolvability(parent_fitness, offspring_fitnesses, evolvability_type, sample_size) for (parent_fitness, offspring_fitnesses, sample_size) in collect(filter(x -> x[3] > 0, zip(fitnesses[current_population_index,:], parent_offspring_fitnesses, sample_sizes)))]
    filtered_sample_sizes = filter(x -> x > 0, sample_sizes)

    return new_population, new_fitnesses, evolvability_observations, filtered_sample_sizes
end

function get_current_population_index(bandit_algorithm::AbstractString, prev_population_index::Int64, K::Int64, maintain_two_populations::Bool, probs::Array{Float64,1})
    if !maintain_two_populations
        current_population_index = 1
    elseif bandit_algorithm == "round robin"
        current_population_index = mod(prev_population_index, K) + 1
    elseif bandit_algorithm == "thompson sampling"
        current_population_index = sample(1:K, WeightVec(probs))
    else
        current_population_index = 1
    end
end



function initialize_population(fitness_function_name::AbstractString = "simple", K::Int64 = 2, N::Int64 = 10)
    if fitness_function_name == "simple"
        IndividualF = SimpleIndividual; FitnessFunctionF = SimpleFitnessFunction
    elseif fitness_function_name == "reisinger"
        IndividualF = ReisingerIndividual; FitnessFunctionF = ReisingerFitnessFunction
    else
        IndividualF = LipsonIndividual; FitnessFunctionF = LipsonFitnessFunction
    end
    populations = [IndividualF() for i in 1:K, j in 1:N]
    fitness_function = FitnessFunctionF()

    fitnesses = [evaluate_fitness(populations[i,j], fitness_function) for i in 1:K, j in 1:N]

    return populations, fitness_function, fitnesses
end

function do_trial(fitness_function_name::AbstractString = "simple", selection_type::AbstractString = "kalman", fitness_evals::Int64 = 10000, N::Int64 = 2, evolvability_type::AbstractString = "variance", heuristic::Int64 = 1, prob_threshold::Float64 = 0.7, N2::Int64 = 1, M::Int64 = 1, q_inference_type::Int64 = 0, bandit_algorithm::AbstractString = "round robin", K::Int64 = 2)

    populations, fitness_function, fitnesses = initialize_population(fitness_function_name, K, N)

    # Initialize filter
    local filter::Union{Kalman, Particle}
    if selection_type == "kalman"
        filter = init_kalman(K)
    elseif selection_type == "particle"
        filter = init_particle(500, K)
    end

    # Initialize bookkeeping
    # For maximum likelihood q inference (TODO, two types of q)
    prev_predictions = [Inf for i in 1:K]
    sum_of_square_diffs_of_predictions = 0.0
    number_of_diffs = 0
    ML_q = 1.
    probs = [1/K for i in 1:K]

    generation = 1
    current_fitness_evals = 0
    fitness_evals_since_selection = 0
    fitness_evals_at_selection = 0
    prev_population_index = K

    results = Float64[]

    while current_fitness_evals < fitness_evals

        delta_fitness_function!(fitness_function)

        if selection_type == "fitness" || (heuristic == 1 && current_fitness_evals / fitness_evals > 0.5) || (heuristic == 2 && fitness_evals_since_selection / (fitness_evals - fitness_evals_at_selection) > 0.1)
            maintain_two_populations = false
        else
            maintain_two_populations = true
        end

        current_population_index = get_current_population_index(bandit_algorithm, prev_population_index, K, maintain_two_populations, probs)

        new_population, new_fitnesses, evolvability_observations, filtered_sample_sizes = tournament_selection(N, N, current_population_index, fitnesses, populations, fitness_function, evolvability_type)
        populations[current_population_index, :] = new_population'
        fitnesses[current_population_index, :] = new_fitnesses'

        current_fitness_evals += N
        fitness_evals_since_selection += N
        if current_fitness_evals > fitness_evals
            break
        end

        selected_for_evolvability = false

        predictions = [Inf for i in 1:K]

        local max_index::Int64
        if selection_type == "point" && mod(generation, M) == 0 && maintain_two_populations
            population_mean_evolvability_observations = zeros(K)
            for current_population_index2 in 1:K
                new_population, new_fitnesses, evolvability_observations, filtered_sample_sizes = tournament_selection(N, N2, current_population_index2, fitnesses, populations, fitness_function, evolvability_type)
                population_mean_evolvability_observations[current_population_index2] = mean(evolvability_observations)
            end

            current_fitness_evals += K * N2
            fitness_evals_since_selection += K * N2
            selected_for_evolvability = true
            max_index = findmax(population_mean_evolvability_observations)[2]
        elseif (selection_type == "kalman" || selection_type == "particle") && maintain_two_populations
            filter_predict(filter, ML_q, q_inference_type, evolvability_type, current_population_index)
            filter_update(filter, evolvability_observations, filtered_sample_sizes, evolvability_type, current_population_index)
            predictions = filter_expected_values(filter)
            probs = filter_probabilities(filter)

            above_threshold = find(x -> x > prob_threshold, probs)
            if length(above_threshold) > 0
                selected_for_evolvability = true
                max_index = above_threshold[1]
                filter_duplicate(filter, max_index)
            end
        end

        # Maximum likelihood estimate of q
        # TODO, two predictions?
        if predictions[1] != Inf
            if prev_predictions[1] != Inf
                sum_of_square_diffs_of_predictions += (predictions[current_population_index] - prev_predictions[current_population_index]).^2
                number_of_diffs += 1
                ML_q = sum_of_square_diffs_of_predictions / number_of_diffs
            end
            prev_predictions = predictions
        end

        if selected_for_evolvability
            # Copy the best population
            for i in 1:K
                for j in 1:N
                    populations[i, j] = copy(populations[max_index, j])
                    fitnesses[i, j] = fitnesses[max_index, j]
                end
                prev_predictions[i] = prev_predictions[max_index]
            end
            fitness_evals_since_selection = 0
            fitness_evals_at_selection = current_fitness_evals
            probs = [1/K for i in 1:K]
        end

        generation += 1
        prev_population_index = current_population_index

        if current_fitness_evals / fitness_evals > 0.9
            push!(results, mean(fitnesses[current_population_index,:]))
        end

    end # End of trial

    return mean(results)
end




# TODO remove max_n, thompson from results string
function do_experiment(job_id::Int64, fitness_function_name::AbstractString, trials::Int64 = 10, selection_type::AbstractString = "kalman", fitness_evals::Int64 = 10000, N::Int64 = 2, evolvability_type::AbstractString = "variance", heuristic::Int64 = 1, P::Float64 = 0.7, N2::Int64 = 1, M::Int64 = 1, q_inference_type::Int64 = 0, bandit_algorithm::AbstractString = "round robin", K::Int64 = 2)
    fitness_results = zeros(trials)

    for trial in 1:trials
        fitness_results[trial] = do_trial(fitness_function_name, selection_type, fitness_evals, N, evolvability_type, heuristic, P, N2, M, q_inference_type, bandit_algorithm, K)
    end

    # Write results to file here
    filename = @sprintf("data/%d.dat", job_id)
    header = @sprintf("fitness_evals-%d+N-%d+evolvability_type-%s+problem-%s+selection_type-%s+N2-%d+M-%d+heuristic-%d+p-%.2f+max_n-%d+q-%d+th-%d\n", fitness_evals, N, evolvability_type, fitness_function_name,selection_type, N2,M,heuristic,P, 0, q_inference_type, 0)
    f = open(filename, "a")
    write(f, header)
    for (datum_i, datum) in enumerate(fitness_results)
        if datum_i != 1
            write(f, ",")
        end
        write(f, @sprintf("%.3f", datum))
    end
    write(f, "\n\n")
    close(f)
end





function do_experiments(job_id::Int64, trials::Int64 = 1, fitness_function_name::AbstractString = "simple")
    fitness_evalss = [5000, 10000] # 5000, 10000?
    Ns = [20, 50] # 100?
    selection_types = ["point","fitness", "kalman", "particle"] # point, fitness, kalman
    evolvability_types = ["std", "maximum"]
    heuristics = [1, 2] # 0, 1, 2
    Ps = [0.6, 0.9] # 0.9
    N2s = [1, 10]
    Ms = [5, 50]
    Ks = [2] # Number of populations

    filename = @sprintf("data/%d.dat", job_id)
    f = open(filename, "w")
    close(f)

    for fitness_evals in fitness_evalss
        for evolvability_type in evolvability_types
            for N in Ns
                for selection_type in selection_types
                    if selection_type == "fitness"
                        do_experiment(job_id, fitness_function_name, trials, selection_type, fitness_evals, N, evolvability_type, 0, 0.0, 0, 0, 0)
                    else
                        for K in Ks
                            for heuristic in heuristics
                                    if selection_type == "point"
                                        for N2 in N2s
                                            N2_adj = N * N2
                                            for M in Ms
                                                do_experiment(job_id, fitness_function_name, trials, selection_type, fitness_evals, N, evolvability_type, heuristic, 0.7, N2_adj, M, 0, "round robin", K)
                                            end
                                        end
                                    else
                                        for P in Ps
                                            if selection_type == "kalman"
                                                q_inference_types = [0, 1] # 0, 1
                                                bandit_algorithms = ["round robin"]
                                            else
                                                q_inference_types = [0, 1, 2] # 0, 1, 2
                                                bandit_algorithms = ["round robin", "thompson sampling"]
                                            end
                                            for q_inference_type in q_inference_types
                                                for bandit_algorithm in bandit_algorithms
                                                    do_experiment(job_id, fitness_function_name, trials, selection_type, fitness_evals, N, evolvability_type, heuristic, P, 0, 0, q_inference_type, bandit_algorithm, K)
                                                end
                                            end
                                        end
                                    end
                            end
                        end
                    end
                end
            end
        end
    end
end


if length(ARGS) > 2
    do_experiments(parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), ARGS[3])
elseif length(ARGS) > 1
    do_experiments(parse(Int64, ARGS[1]), parse(Int64, ARGS[2]))
elseif length(ARGS) > 0
    do_experiments(parse(Int64, ARGS[1]))
else
    println("Provide job ID")
end



