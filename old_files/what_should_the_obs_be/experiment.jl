#=
For now, the only problem type is the simple problem.
All other parameters are varied as before

TODOS:
- Distribute jobs
- Plot results

Once it's the same as python version, and faster
- Infer process noise magnitude
- Implement noise distribution for observing the maximum fitness of offspring
=#

# TODO make these modules
include("kalman.jl")
include("particle.jl")
using Distributions

#=
# Simple model
type Individual
    f::Float64
    e::Float64
end

import Base.copy
Base.copy(i::Individual) = Individual(i.f, i.e)

function new_individual()
    return Individual(0.0, 10.0)
end

function mutated_copy(p::Individual)

    q = 1.0
    f2 = p.f+ randn() * p.e
    e2 = p.e + randn() * sqrt(q)
    if e2 < 0
        e2 = 0.0001
    end
    return Individual(f2, e2)
end

function evaluate_fitness(individual::Individual)
    return individual.f
end
=#



# Reisinger
type Individual
    bitstring::Array{Int64,1}
    mutation_mask::Array{Int64,1}
end

import Base.copy
Base.copy(i::Individual) = Individual(copy(i.bitstring), copy(i,mutation_mask))

function new_individual()
    return Individual(rand(Distributions.Bernoulli(0.5), 100), rand(Distributions.Bernoulli(0.5), 100))
end

function mutated_copy(individual::Individual, target::Bool)
    mutation_rate = 0.01
    if target
        mutation_rate *= 0.1
    end

    mutations = rand(Distributions.Bernoulli(mutation_rate), 100)
    masked_mutations = mutations & individual.mutation_mask
    new_bitstring = individual.bitstring $ masked_mutations
    meta_mutations = rand(Distributions.Bernoulli(0.5*mutation_rate), 100)
    new_mutation_mask = individual.mutation_mask $ meta_mutations
    return Individual(new_bitstring, new_mutation_mask)
end

function evaluate_fitness(individual::Individual, target::Individual)
    return mean(individual.bitstring .== target.bitstring)
end


function observe_evolvability(parent_fitnesses::Array{Float64,1}, fitnesses::Array{Float64, 2}, evolvability_type::ASCIIString)
    evolvability_observations = zeros(2)
    if evolvability_type == "variance"
        evolvability_observations = squeeze(var(fitnesses, 2), 2)
    elseif evolvability_type == "std"
        evolvability_observations = squeeze(std(fitnesses, 2), 2)
    else
        fitness_diffs = fitnesses - repmat(parent_fitnesses, 1, size(fitnesses)[2])
        evolvability_observations = squeeze(maximum(fitness_diffs,2),2)
    end

    return evolvability_observations
end

function do_trial(job_id::Int64, selection_type::ASCIIString = "kalman", fitness_evals::Int64 = 10000, N::Int64 = 2, evolvability_type::ASCIIString = "variance", heuristic::Bool = false, prob_threshold::Float64 = 0.7, N2::Int64 = 1, M::Int64 = 1)

    parents = [new_individual() for i in 1:2]
    target = new_individual()
    # TODO temp
    evaluate_fitness2(individual::Individual) = evaluate_fitness(individual::Individual, target::Individual)
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

    while current_fitness_evals < fitness_evals

        target = mutated_copy(target, true)

        offspring_populations = [mutated_copy(parent, false) for parent in parents, n in 1:N]
        evaluate_fitness2(individual::Individual) = evaluate_fitness(individual::Individual, target::Individual)
        offspring_fitnesses = map(evaluate_fitness2, offspring_populations)

        maintain_two_populations = false
        if selection_type == "point" && (!heuristic || current_fitness_evals < div(fitness_evals, 2))
            maintain_two_populations = true
        elseif selection_type == "kalman" && (!heuristic || current_fitness_evals < div(fitness_evals, 2))
            maintain_two_populations = true
        elseif selection_type == "particle" && (!heuristic || current_fitness_evals < div(fitness_evals, 2))
            maintain_two_populations = true
        end

        if maintain_two_populations
            current_fitness_evals += 2N
        else
            current_fitness_evals += N
        end

        evolvability_observations = observe_evolvability(parent_fitnesses, offspring_fitnesses, evolvability_type)::Array{Float64,1}

        # Select for fitness
        # TODO findmax gets us the indices as they were in the whole offspring_fitnesses array. What I'm doing here is messy... use something like ravel() from numpy?
        max_indices = map(x-> mod(x, 2) == 0 ? div(x, 2) : div(x+1, 2), squeeze(findmax(offspring_fitnesses, 2)[2], 2))

        parents = [offspring_populations[i, max_indices[i]]::Individual for i in 1:2 ]
        parent_fitnesses = [offspring_fitnesses[i, max_indices[i]]::Float64 for i in 1:2]

        if current_fitness_evals > fitness_evals
            break
        end

        selected_for_evolvability = false

        local max_index::Int64
        if selection_type == "point" && mod(generation, M) == 0 && maintain_two_populations
            evolvability_selection_populations = [mutated_copy(parent, false) for parent in parents, n in 1:N2]
            evaluate_fitness2(individual::Individual) = evaluate_fitness(individual::Individual, target::Individual)
            evolvability_selection_fitnesses = map(evaluate_fitness2, evolvability_selection_populations)
            evolvability_observations = observe_evolvability(parent_fitnesses, evolvability_selection_fitnesses, evolvability_type)

            current_fitness_evals += 2N2
            selected_for_evolvability = true
            max_index = findmax(evolvability_observations)[2]

        elseif selection_type == "kalman" && maintain_two_populations
            kalman_obs(kalman, evolvability_observations, N, evolvability_type)

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
            particle_pred(particle)
            particle_update(particle, evolvability_observations, N, evolvability_type)
            particle_resample(particle)
            frac = sum((particle.x2s .> particle.x1s) .* exp(particle.weights))
            if frac > prob_threshold || 1-frac > prob_threshold
                selected_for_evolvability = true
                max_index = round(Int64, frac) + 1
                particle_duplicate(particle, max_index)
            end
        end

        if selected_for_evolvability
            parent = parents[max_index]
            parents = [mutated_copy(parent, false) for i in 1:2]
        end

        generation += 1

    end # End of trial

    return parent_fitnesses[1]
end

function do_experiment(job_id::Int64, trials::Int64 = 10, selection_type::ASCIIString = "kalman", fitness_evals::Int64 = 10000, N::Int64 = 2, evolvability_type::ASCIIString = "variance", heuristic::Bool = false, P::Float64 = 0.7, N2::Int64 = 1, M::Int64 = 1)
    fitness_results = zeros(trials)

    for trial in 1:trials
        fitness_results[trial] = do_trial(job_id, selection_type, fitness_evals, N, evolvability_type, heuristic, P, N2, M)
    end

    # Write results to file here
    filename = @sprintf("data/%d.dat", job_id)
    header = @sprintf("fitness_evals-%d+N-%d+evolvability_type-%s+problem-simple+selection_type-%s+N2-%d+M-%d+heuristic-%d+p-%.2f\n", fitness_evals, N, evolvability_type,selection_type, N2,M,heuristic,P)
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

function do_experiments(job_id::Int64, trials::Int64 = 1)
    fitness_evalss = [5000, 10000]
    Ns = [2, 20]
    selection_types = ["fitness", "point", "kalman", "particle"]
    evolvability_types = ["std", "maximum"]
    heuristics = [false, true]
    Ps = [0.6, 0.75, 0.9]
    N2s = [1, 10]
    Ms = [5, 50]

    filename = @sprintf("data/%d.dat", job_id)
    f = open(filename, "w")
    close(f)

    for fitness_evals in fitness_evalss
        for evolvability_type in evolvability_types
            for N in Ns
                for selection_type in selection_types
                    if selection_type == "fitness"
                        do_experiment(job_id, trials, selection_type, fitness_evals, N, evolvability_type, false, 0.0, 0, 0)
                    else
                        for heuristic in heuristics
                            if selection_type == "point"
                                for N2 in N2s
                                    N2_adj = N * N2
                                    for M in Ms
                                        do_experiment(job_id, trials, selection_type, fitness_evals, N, evolvability_type, heuristic, 0.7, N2_adj, M)
                                    end
                                end
                            else
                                for P in Ps
                                    do_experiment(job_id, trials, selection_type, fitness_evals, N, evolvability_type, heuristic, P, 0, 0)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

end


if size(ARGS)[1] > 1
    do_experiments(parse(Int64, ARGS[1]), parse(Int64, ARGS[2]))
elseif size(ARGS)[1] > 0
    do_experiments(parse(Int64, ARGS[1]))
else
    println("Provide job ID")
end



