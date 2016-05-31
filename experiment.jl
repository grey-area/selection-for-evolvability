include("kalman.jl")
include("particle.jl")
include("fitnessfunctions.jl")

function observe_evolvability(parent_fitnesses::Array{Float64,1}, fitnesses::Array{Float64, 2}, evolvability_type::ASCIIString, n::Int64, max_n::Int64)
    evolvability_observations = zeros(2)
    if evolvability_type == "variance"
        evolvability_observations = var(fitnesses, 2)
    elseif evolvability_type == "std"
        # unbiased standard deviation
        correction_term = sqrt(2.0/(n-1)) * gamma(n/2) / gamma((n-1)/2)

        evolvability_observations = std(fitnesses, 2) / correction_term
    else
        fitness_diffs = fitnesses - repmat(parent_fitnesses, 1, size(fitnesses)[2])
        if max_n == -1 || max_n == n
            evolvability_observations = maximum(fitness_diffs, 2)
        else
            num_rows = div(size(fitnesses)[2], max_n)
            evolvability_observations = zeros(2, num_rows)
            for i in 1:num_rows
                j = (i-1)*max_n + 1
                evolvability_observations[:, i] = maximum(fitness_diffs[:, j:j+max_n-1], 2)
            end
        end
    end

    return evolvability_observations
end

function do_trial{T<:Union{SimpleIndividual, ReisingerIndividual, LipsonIndividual}}(job_id::Int64, parent::T, fitness_function::FitnessFunction, selection_type::ASCIIString = "kalman", fitness_evals::Int64 = 10000, N::Int64 = 2, evolvability_type::ASCIIString = "variance", heuristic::Int64 = 1, prob_threshold::Float64 = 0.7, N2::Int64 = 1, M::Int64 = 1, max_n::Int64 = -1, q_inference_type::Int64 = 0, thompson::Bool = false)

    parents::Array{T, 1}
    parents = T[mutated_copy(parent) for i in 1:2]

    # For maximum likelihood q inference
    prev_prediction1 = Inf
    prev_prediction2 = Inf
    sum_of_square_diffs_of_predictions = 0.0
    number_of_diffs = 0
    ML_q = 1.

    sample_size = N
    if max_n != -1 && evolvability_type == "maximum"
        sample_size = max_n
    end

    # TODO check types are being preserved
    evaluate_fitness2(individual::T) = evaluate_fitness(individual, fitness_function)
    parent_fitnesses = map(evaluate_fitness2, parents)

    local kalman::Kalman
    local particle::Particle
    if selection_type == "kalman"
        kalman = init_kalman()
    elseif selection_type == "particle"
        particle = init_particle(500)
    end

    generation = 1
    current_fitness_evals = 0
    fitness_evals_since_selection = 0
    fitness_evals_at_selection = 0

    results = Float64[]

    frac = 0.5

    while current_fitness_evals < fitness_evals

        delta_fitness_function!(fitness_function)

        # TODO check types are being preverved
        offspring_populations = Array(T, 2, N)
        offspring_fitnesses = Array(Float64, 2, N)
        for i in 1:2, j in 1:N
            new_individual = mutated_copy(parents[i])
            offspring_populations[i, j] = new_individual
            offspring_fitnesses[i, j] = evaluate_fitness(new_individual, fitness_function)
        end

        if selection_type == "fitness" || (heuristic == 1 && current_fitness_evals / fitness_evals > 0.5) || (heuristic == 2 && fitness_evals_since_selection / (fitness_evals - fitness_evals_at_selection) > 0.1)
            maintain_two_populations = false
        else
            maintain_two_populations = true
        end

        if maintain_two_populations && !thompson
            current_fitness_evals += 2N
            fitness_evals_since_selection += 2N
        else
            current_fitness_evals += N
            fitness_evals_since_selection += N
        end

        evolvability_observations = observe_evolvability(parent_fitnesses, offspring_fitnesses, evolvability_type, N, max_n)::Array{Float64,2}

        # TODO types of selection other than select only max

        # Select for fitness
        max_indices = map(x-> mod(x, 2) == 0 ? div(x, 2) : div(x+1, 2), squeeze(findmax(offspring_fitnesses, 2)[2], 2))

        thompson_i = 0
        if thompson
            if rand() < frac
                thompson_i = 2
            else
                thompson_i = 1
            end
        end

        if !thompson
            parents = [offspring_populations[i, max_indices[i]]::T for i in 1:2 ]
            parent_fitnesses = [offspring_fitnesses[i, max_indices[i]]::Float64 for i in 1:2]
        else
            parents[thompson_i] = offspring_populations[thompson_i, max_indices[thompson_i]]
            parent_fitnesses[thompson_i] = offspring_fitnesses[thompson_i, max_indices[thompson_i]]
        end


        if current_fitness_evals > fitness_evals
            break
        end

        selected_for_evolvability = false

        prediction1 = Inf
        prediction2 = Inf

        local max_index::Int64
        if selection_type == "point" && mod(generation, M) == 0 && maintain_two_populations

            evolvability_selection_populations = Array(T, 2, N2)
            evolvability_selection_fitnesses = Array(Float64, 2, N2)
            for i in 1:2, j in 1:N2
                new_individual = mutated_copy(parents[i])
                evolvability_selection_populations[i, j] = new_individual
                evolvability_selection_fitnesses[i, j] = evaluate_fitness(new_individual, fitness_function)
            end

            evolvability_observations = squeeze(observe_evolvability(parent_fitnesses, evolvability_selection_fitnesses, evolvability_type, N2, -1)::Array{Float64,2}, 2)

            current_fitness_evals += 2N2
            fitness_evals_since_selection += 2N2
            selected_for_evolvability = true
            max_index = findmax(evolvability_observations)[2]

        elseif selection_type == "kalman" && maintain_two_populations
            kalman_pred(kalman, ML_q, q_inference_type)
            kalman_update(kalman, evolvability_observations, sample_size, evolvability_type)
            prediction1 = kalman.xs[1]
            prediction2 = kalman.xs[2]

            # frac our belief that x2 > x1
            c1 = kalman.ps[1,1]; c2 = kalman.ps[1,2]; c3 = kalman.ps[2,2]
            denominator_term = c1 - 2c2 + c3
            if denominator_term < 0.0
                denominator_term = 0.0
            end
            denominator = √(2.0 * denominator_term)
            numerator = kalman.xs[2] - kalman.xs[1]
            frac = 0.5 * (1.0 + erf(numerator / denominator))

            if frac > prob_threshold || 1-frac > prob_threshold
                selected_for_evolvability = true
                max_index = round(Int64, frac) + 1
                kalman_duplicate(kalman, max_index)
            end
        elseif selection_type == "particle" && maintain_two_populations
            particle_pred(particle, ML_q, q_inference_type, thompson_i)
            particle_update(particle, evolvability_observations, sample_size, evolvability_type, thompson_i)
            particle_resample(particle)
            prediction1 = sum(particle.x1s .* exp(particle.weights))
            prediction2 = sum(particle.x2s .* exp(particle.weights))

            frac = sum((particle.x2s .> particle.x1s) .* exp(particle.weights))
            if frac > prob_threshold || 1-frac > prob_threshold
                selected_for_evolvability = true
                max_index = round(Int64, frac) + 1
                frac = 0.5
                particle_duplicate(particle, max_index)
            end
        end

        if prediction1 != Inf
            if prev_prediction1 != Inf
                sum_of_square_diffs_of_predictions += (prediction1 - prev_prediction1)^2
                sum_of_square_diffs_of_predictions += (prediction2 - prev_prediction2)^2
                number_of_diffs += 2
                ML_q = sum_of_square_diffs_of_predictions / number_of_diffs
            end
            prev_prediction1 = prediction1
            prev_prediction2 = prediction2
        end

        if selected_for_evolvability
            parent = parents[max_index]
            parents = [mutated_copy(parent) for i in 1:2]
            prediction1 = Inf
            prediction2 = Inf
            prev_prediction1 = Inf
            prev_prediction2 = Inf
            fitness_evals_since_selection = 0
            fitness_evals_at_selection = current_fitness_evals
        end

        generation += 1

        if current_fitness_evals / fitness_evals > 0.9
            push!(results, parent_fitnesses[1])
        end

    end # End of trial

    return mean(parent_fitnesses[1])
end

function do_experiment(job_id::Int64, fitness_function_name::AbstractString, trials::Int64 = 10, selection_type::ASCIIString = "kalman", fitness_evals::Int64 = 10000, N::Int64 = 2, evolvability_type::ASCIIString = "variance", heuristic::Int64 = 1, P::Float64 = 0.7, N2::Int64 = 1, M::Int64 = 1, max_n::Int64 = -1, q_inference_type::Int64 = 0, thompson::Bool = false)
    fitness_results = zeros(trials)

    for trial in 1:trials

        fitness_function::FitnessFunction

        if fitness_function_name == "simple"
            parent = SimpleIndividual()
            fitness_function = SimpleFitnessFunction()
        elseif fitness_function_name == "reisinger"
            parent = ReisingerIndividual()
            fitness_function = ReisingerFitnessFunction()
        else
            parent = LipsonIndividual()
            fitness_function = LipsonFitnessFunction()
        end

        fitness_results[trial] = do_trial(job_id, parent, fitness_function, selection_type, fitness_evals, N, evolvability_type, heuristic, P, N2, M, max_n, q_inference_type, thompson)
    end

    # Write results to file here
    filename = @sprintf("data/%d.dat", job_id)
    header = @sprintf("fitness_evals-%d+N-%d+evolvability_type-%s+problem-%s+selection_type-%s+N2-%d+M-%d+heuristic-%d+p-%.2f+max_n-%d+q-%d+th-%d\n", fitness_evals, N, evolvability_type, fitness_function_name,selection_type, N2,M,heuristic,P, max_n, q_inference_type, thompson)
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

# trials, selection type, fitness evals, N, evolvability type
#@code_warntype do_experiment(10, "point_evolvability", 30000, 2, "variance")
#do_experiment(10, "kalman", 30000, 2, "variance")

function do_experiments(job_id::Int64, trials::Int64 = 1, fitness_function_name::AbstractString = "simple")
    fitness_evalss = [5000, 10000] # 5000, 10000?
    Ns = [2, 50] # 100?
    selection_types = ["fitness", "kalman", "particle"] # point
    evolvability_types = ["std", "maximum"]
    heuristics = [1, 2] # 0, 1, 2
    Ps = [0.6, 0.9]
    N2s = [1, 10]
    Ms = [5, 50]
    max_ns = [2, -1]

    filename = @sprintf("data/%d.dat", job_id)
    f = open(filename, "w")
    close(f)

    for fitness_evals in fitness_evalss
        for evolvability_type in evolvability_types
            for N in Ns
                for selection_type in selection_types
                    if selection_type == "fitness"
                        do_experiment(job_id, fitness_function_name, trials, selection_type, fitness_evals, N, evolvability_type, 0, 0.0, 0, 0, -1, 0)
                    else
                        for heuristic in heuristics
                            for max_n in max_ns
                                if selection_type == "point"
                                    for N2 in N2s
                                        N2_adj = N * N2
                                        for M in Ms
                                            do_experiment(job_id, fitness_function_name, trials, selection_type, fitness_evals, N, evolvability_type, heuristic, 0.7, N2_adj, M, max_n, 0)
                                        end
                                    end
                                else
                                    for P in Ps
                                        if selection_type == "kalman"
                                            q_inference_types = [0, 1] # 0, 1
                                            thompsons = [false]
                                        else
                                            q_inference_types = [0, 1, 2] # 0, 1, 2
                                            thompsons = [false, true]
                                        end
                                        for q_inference_type in q_inference_types
                                            for thompson in thompsons
                                                do_experiment(job_id, fitness_function_name, trials, selection_type, fitness_evals, N, evolvability_type, heuristic, P, 0, 0, max_n, q_inference_type, thompson)
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



