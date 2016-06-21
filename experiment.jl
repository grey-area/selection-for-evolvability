include("kalman.jl")
include("particle.jl")
include("fitnessfunctions.jl")

#=
TODO list
- Check PF can track maximum properly
- Don't just return the mean of trial results?
- Calculate and return evolvability
- Bandits
=#

using Iterators

# Calculate an unbiased estimate of evolvability
function observe_evolvability(parent_fitness::Float64, offspring_fitnesses::Array{Float64, 1}, evolvability_type::AbstractString)
    if evolvability_type == "variance"
        evolvability_observation = var(offspring_fitnesses)
    elseif evolvability_type == "std"
        sample_size = size(offspring_fitnesses)[1]
        correction_term = sqrt(2.0/(sample_size-1)) * gamma(sample_size/2) / gamma((sample_size-1)/2)
        evolvability_observation = std(offspring_fitnesses) / correction_term
    else
        fitness_differences = offspring_fitnesses - parent_fitness
        evolvability_observation = maximum(fitness_differences)
    end
    return evolvability_observation::Float64
end

function crossover_population{T<:Union{SimpleIndividual, ReisingerIndividual, TurneyIndividual, LipsonIndividual}}(crossover_rate::Float64, fitness_function::FitnessFunction, new_population::Array{T, 1}, new_fitnesses::Array{Float64, 1}, N2::Int64)

    if rand() < crossover_rate && fitness_function.implements_crossover
        shuffle!(new_population)
        crossed_population = Array{T}(N2)

        for i in 0:(div(N2, 2) - 1)
            base_index = 2i + 1

            c1, c2 = crossover(new_population[base_index], new_population[base_index+1])
            crossed_population[base_index] = c1
            crossed_population[base_index+1] = c2
        end

        new_fitnesses = Float64[evaluate_fitness(individual, fitness_function) for individual in crossed_population]
        new_population = crossed_population
    end

    return new_population, new_fitnesses
end

# Select and mutate using tournament selection for one of the populations
# return the new population and fitnesses
# calculate evolvability observations for the population in question
function tournament_selection{T<:Union{SimpleIndividual, ReisingerIndividual, TurneyIndividual, LipsonIndividual}}(N::Int64, N2::Int64, current_population_index::Int64, fitnesses::Array{Float64, 2}, populations::Array{T, 2}, fitness_function::FitnessFunction, evolvability_type::AbstractString, tournament_size::Int64)
    # array for new population, array of their fitnesses, array of arrays of fitnesses, array of sample sizes

    new_population = Array{T}(N2)
    new_fitnesses = zeros(N2)
    evolvability_observations = Array{Float64, 1}(0)
    filtered_sample_sizes = Array{Int64, 1}(0)
    parent_offspring_fitnesses = [Array{Float64}(0) for i in 1:N]

    for i in 0:(div(N2, 2) - 1)
        contender_indices = sample(1:N, tournament_size, replace=false)
        fitter_index = contender_indices[findmax(squeeze(fitnesses[current_population_index, contender_indices], 1))[2]]

        base_index = 2i + 1
        for j in 0:1
            new_index = base_index + j
            new_population[new_index] = mutated_copy(populations[current_population_index, fitter_index])
            new_fitnesses[new_index] = evaluate_fitness(new_population[new_index], fitness_function)

            if evolvability_type != "maximum"
                push!(parent_offspring_fitnesses[fitter_index], new_fitnesses[new_index])
            end
        end

        if evolvability_type == "maximum"
            new_evolvability = observe_evolvability(fitnesses[current_population_index, fitter_index], new_fitnesses[base_index:base_index+1], evolvability_type)
            push!(evolvability_observations, new_evolvability)
            push!(filtered_sample_sizes, 2)
        end
    end

    if evolvability_type != "maximum"
        sample_sizes = Int64[length(fitnesses) for fitnesses in parent_offspring_fitnesses]
        # Now, for each non-zero entry in sample sizes, we need an evolvability observation
        evolvability_observations = Float64[observe_evolvability(parent_fitness, offspring_fitnesses, evolvability_type) for (parent_fitness, offspring_fitnesses, sample_size) in collect(filter(x -> x[3] > 0, zip(fitnesses[current_population_index,:], parent_offspring_fitnesses, sample_sizes)))]
        filtered_sample_sizes = filter(x -> x > 0, sample_sizes)
    end

    return new_population, new_fitnesses, evolvability_observations, filtered_sample_sizes
end

# Which population goes through a generation next?
# If we aren't maintaining multiple populations, then it's always population 1
# Otherwise, it might be round robin or according to a bandit algorithm
function get_current_population_index(bandit_algorithm::AbstractString, prev_population_index::Int64, K::Int64, maintain_multiple_populations::Bool, probs::Array{Float64,1})

    if !maintain_multiple_populations
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
    elseif fitness_function_name == "turney"
        IndividualF = TurneyIndividual; FitnessFunctionF = TurneyFitnessFunction
    else
        IndividualF = LipsonIndividual; FitnessFunctionF = LipsonFitnessFunction
    end
    populations = [IndividualF() for i in 1:K, j in 1:N]
    fitness_function = FitnessFunctionF()

    fitnesses = [evaluate_fitness(populations[i,j], fitness_function) for i in 1:K, j in 1:N]

    return populations, fitness_function, fitnesses
end


function evolvability_point_estimates{T<:Union{SimpleIndividual, ReisingerIndividual, TurneyIndividual, LipsonIndividual}}(generation::Int64, K::Int64, M::Int64, N::Int64, N2::Int64, fitnesses::Array{Float64, 2}, populations::Array{T, 2}, fitness_function::FitnessFunction, evolvability_type::AbstractString, current_fitness_evals::Int64, fitness_evals_since_selection::Int64, tournament_size::Int64)
    max_index = 0

    if mod(generation, M) == 0
        population_mean_evolvability_observations = zeros(K)
        for current_population_index2 in 1:K
            new_population, new_fitnesses, evolvability_observations, sample_sizes = tournament_selection(N, N2, current_population_index2, fitnesses, populations, fitness_function, evolvability_type, tournament_size)
            population_mean_evolvability_observations[current_population_index2] = mean(evolvability_observations)
        end

        current_fitness_evals += K * N2
        fitness_evals_since_selection += K * N2

        max_index = findmax(population_mean_evolvability_observations)[2]
    end

    return max_index, current_fitness_evals, fitness_evals_since_selection
end


function update_filter(evolvability_type::AbstractString, filter::Union{Kalman, Particle}, ML_q::Float64, q_inference_type::AbstractString, current_population_index::Int64, prob_threshold::Float64, evolvability_observations::Array{Float64, 1}, sample_sizes::Array{Int64, 1})
    max_index = 0

    filter_predict(filter, ML_q, q_inference_type, evolvability_type, current_population_index)
    filter_update(filter, evolvability_observations, sample_sizes, evolvability_type, current_population_index)
    predictions = filter_expected_values(filter)
    probs = filter_probabilities(filter)

    above_threshold = find(x -> x > prob_threshold, probs)
    if length(above_threshold) > 0
        max_index = above_threshold[1]
        filter_duplicate(filter, max_index)
    end

    return max_index, predictions, probs
end


# Duplicate one population
function select_for_evolvability{T<:Union{SimpleIndividual, ReisingerIndividual, TurneyIndividual, LipsonIndividual}}(populations::Array{T, 2}, fitnesses::Array{Float64, 2}, max_index::Int64, prev_predictions::Array{Float64, 1}, current_fitness_evals::Int64, fitness_evals_since_selection::Int64, fitness_evals_at_selection::Int64, K::Int64, N::Int64)
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

    return fitness_evals_since_selection, fitness_evals_at_selection, probs
end






function do_trial(fitness_function_name::AbstractString = "simple", selection_type::AbstractString = "kalman", fitness_evals::Int64 = 10000, N::Int64 = 2, evolvability_type::AbstractString = "variance", termination_heuristic::AbstractString = "halfway", prob_threshold::Float64 = 0.7, N2::Int64 = 1, M::Int64 = 1, q_inference_type::AbstractString = "none", bandit_algorithm::AbstractString = "round robin", K::Int64 = 2, tournament_size::Int64 = 2, problem_delta_rate::Float64 = 1.0)

    # Initial populations and fitnesses, K by N arrays
    populations, fitness_function, fitnesses = initialize_population(fitness_function_name, K, N)
    populations::Array{Union{LipsonIndividual, SimpleIndividual, ReisingerIndividual, TurneyIndividual}}

    # Initialize filter
    local filter::Union{Kalman, Particle}
    if selection_type == "particle"
        filter = init_particle(500, K)
    else
        filter = init_kalman(K)
    end

    # Maximum likelihood process noise inference bookkeeping
    prev_predictions = [Inf for i in 1:K]
    sum_of_square_diffs_of_predictions = 0.0
    number_of_diffs = 0
    ML_q = 1.
    probs = [1/K for i in 1:K]

    # More bookkeeping
    generation = 1
    current_fitness_evals = 0
    fitness_evals_since_selection = 0
    fitness_evals_at_selection = 0
    prev_population_index = K

    results = Float64[]

    while current_fitness_evals < fitness_evals

        # For fitness functions that change over time
        delta_fitness_function!(fitness_function, problem_delta_rate)

        # Determines whether we maintain multiple populations
        # We only maintain multiple populations if we are not selecting for fitness alone and the termination criterion isn't met.
        # Termination criterion 1 is met when half of the fitness evaluations are used up
        # Termination criterion 2 is met when the number of fitness evaluations since we last selected for evolvability is more than 10% of the fitness evaluations remaining
        maintain_multiple_populations = true
        if selection_type == "fitness" || (termination_heuristic == "halfway" && current_fitness_evals / fitness_evals > 0.5) || (termination_heuristic == "relative" && fitness_evals_since_selection / (fitness_evals - fitness_evals_at_selection) > 0.1)
            maintain_multiple_populations = false
        end

        # Determines which population goes through a generation next
        # If we're not maintaining multiple populations, then it will always be population 1
        # Otherwise, the populations might be handled round robin or chosen by a bandit algorithm
        current_population_index = get_current_population_index(bandit_algorithm, prev_population_index, K, maintain_multiple_populations, probs)

        # For the population chosen, go through one generation of selection and mutation
        # Also record the evolvabilities of the parents, and the sample sizes used to calculate those evolvabilities
        new_population, new_fitnesses, evolvability_observations, filtered_sample_sizes = tournament_selection(N, N, current_population_index, fitnesses, populations, fitness_function, evolvability_type, tournament_size)
        crossover_rate = 1.0
        new_population, new_fitnesses = crossover_population(crossover_rate, fitness_function, new_population, new_fitnesses, N)
        populations[current_population_index, :] = new_population'
        fitnesses[current_population_index, :] = new_fitnesses'

        # Increment fitness evaluation counts
        current_fitness_evals += N
        fitness_evals_since_selection += N
        if current_fitness_evals > fitness_evals
            break
        end

        # Determine whether to select for evolvability and do so
        if maintain_multiple_populations
            # If it's time to calculate evolvability point estimates, do so, and determine which population has the larger evolvability (for selection)
            if selection_type == "point"
                max_index, current_fitness_evals, fitness_evals_since_selection = evolvability_point_estimates(generation, K, M, N, N2, fitnesses, populations, fitness_function, evolvability_type, current_fitness_evals, fitness_evals_since_selection, tournament_size)

            # Update the filter (predict then update step)
            # Calculate the predictions of the filter and the probability that each population is fittest
            # If a probability threshold is crossed, choose a population to select for evolvability
            elseif selection_type in ["kalman", "particle"]
                max_index, predictions, probs = update_filter(evolvability_type, filter, ML_q, q_inference_type, current_population_index, prob_threshold, evolvability_observations, filtered_sample_sizes)

                # Maximum likelihood estimate of process noise.
                if prev_predictions[current_population_index] != Inf
                    sum_of_square_diffs_of_predictions += (predictions[current_population_index] - prev_predictions[current_population_index]).^2
                    number_of_diffs += 1
                    ML_q = sum_of_square_diffs_of_predictions / number_of_diffs
                end
                prev_predictions = predictions
            end

            # If we are selecting for evolvability, do so
            if max_index > 0
                fitness_evals_since_selection, fitness_evals_at_selection, probs = select_for_evolvability(populations, fitnesses, max_index, prev_predictions, current_fitness_evals, fitness_evals_since_selection, fitness_evals_at_selection, K, N)
            end
        end

        generation += 1
        prev_population_index = current_population_index

        # Record mean fitness during the last 10% of the run
        if current_fitness_evals / fitness_evals > 0.9
            push!(results, mean(fitnesses[current_population_index,:]))
        end

    end # End of trial

    evolvability = 0.0
    # Return mean of mean fitness during last 10% of the run
    return mean(results), evolvability
end





# Run each trial, record the results to a file identified by the job id
function do_experiment(fitness_function_name::AbstractString, trials::Int64 = 10, selection_type::AbstractString = "kalman", fitness_evals::Int64 = 10000, N::Int64 = 2, evolvability_type::AbstractString = "variance", termination_heuristic::AbstractString = "halfway", P::Float64 = 0.7, N2::Int64 = 1, M::Int64 = 1, q_inference_type::AbstractString = "none", bandit_algorithm::AbstractString = "round robin", K::Int64 = 2, tournament_size::Int64 = 2, problem_delta_rate::Float64 = 1.0)
    fitness_results = zeros(trials)
    evolvability_results = zeros(trials)

    for trial in 1:trials
        (fitness_results[trial], evolvability_results[trial]) = do_trial(fitness_function_name, selection_type, fitness_evals, N, evolvability_type, termination_heuristic, P, N2, M, q_inference_type, bandit_algorithm, K, tournament_size, problem_delta_rate)
    end

    return fitness_results, evolvability_results
end


#Iterate over combinations of parameter settings; run experiments
function do_experiments(job_id::Int64, trials::Int64 = 1)
    filename = @sprintf("data/%d.dat", job_id)
    f = open(filename, "w")
    close(f)

    fitness_function_name = rand(["simple", "reisinger", "turney", "lipson"])
    fitness_evals         = rand(1000:500:10000)
    N                     = rand(2:2:100)
    tournament_size       = rand(2:2:N)
    problem_delta_rate    = rand() * 2

    (base_fitness_results, base_evolvability_results) = do_experiment(fitness_function_name, trials, "fitness", fitness_evals, N, "std", "none", 0.0, 0, 0, "none", "round robin", 1, tournament_size, problem_delta_rate)

    for selection_type in ["point", "kalman", "particle"]

        evolvability_type = rand(["std", "maximum"])
        termination_heuristic = rand(["none", "halfway", "relative"])
        K = rand(2:5)
        N2 = rand(1:10)
        M = rand(5:50)
        P = 0.6 + 0.4 * rand()
        q_inference_type = rand(["ML", "mean_posterior"])
        bandit_algorithm = "round robin"

        (fitness_results, evolvability_results) = do_experiment(fitness_function_name, trials, selection_type, fitness_evals, N, evolvability_type, termination_heuristic, P, N2*N, M, q_inference_type, bandit_algorithm, K, tournament_size, problem_delta_rate)
        relative_fitness_results = fitness_results ./ base_fitness_results - 1.0
        relative_evolvability_results = evolvability_results ./ base_evolvability_results - 1.0

        result_string = @sprintf("%s, %d, %d, %d, %f, %s, %s, %s, %d, %d, %d, %f, %s, %s, %f\n", fitness_function_name, fitness_evals, N, tournament_size, problem_delta_rate, selection_type, evolvability_type, termination_heuristic, K, N2, M, P, q_inference_type, bandit_algorithm, mean(relative_fitness_results))
        f = open(filename, "a")
        write(f, result_string)
        close(f)

    end # End of selection type loop
end


if length(ARGS) > 1
    do_experiments(parse(Int64, ARGS[1]), parse(Int64, ARGS[2]))
elseif length(ARGS) > 0
    do_experiments(parse(Int64, ARGS[1]))
else
    println("Provide job ID, and optionally the number of trials")
end



