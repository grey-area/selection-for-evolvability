include("experiment.jl")
println("")

using FactCheck

facts("observe_evolvability()") do
    epsilon = 0.03
    N = 100000

    for sigma in [1.0, 2.0]
        results1 = zeros(N)
        results2 = zeros(N)
        results3 = zeros(N)
        for i in 1:N
            offspring_fitnesses = sigma * randn(2)
            results1[i] = observe_evolvability(1., offspring_fitnesses, "std")
            results2[i] = observe_evolvability(1., offspring_fitnesses, "maximum")
            results3[i] = observe_evolvability(1., offspring_fitnesses, "variance")
        end

        @fact mean(results1) --> roughly(sigma, epsilon)
        @fact mean(results2) --> roughly(0.565229237sigma - 1.0, epsilon)
        @fact mean(results3) --> roughly(sigma^2, epsilon)
    end
end

facts("get_current_population_index()") do
    K = 4
    probs = [1/K for i in 1:K]

    for bandit_algorithm in ["round robin", "thompson sampling"]
        for k in 1:K
            @fact get_current_population_index(bandit_algorithm, k, K, false, probs) --> 1
        end
    end

    @fact get_current_population_index("round robin", 1, K, true, probs) --> 2
    @fact get_current_population_index("round robin", 2, K, true, probs) --> 3
    @fact get_current_population_index("round robin", 3, K, true, probs) --> 4
    @fact get_current_population_index("round robin", 4, K, true, probs) --> 1

    K = 3
    N = 100000
    probs = [0.1, 0.5, 0.4]
    results = zeros(3)
    prev_j = 1
    for i in 1:N
        j = get_current_population_index("thompson sampling", prev_j, K, true, probs)
        prev_j = j
        results[j] += 1
    end
    @fact results/N --> roughly(probs, 0.01)
end

facts("select_for_evolvability()") do
    K = 3
    N = 2

    for k in 1:K

        populations = [SimpleIndividual(rand(), rand()) for i in 1:K, j in 1:N]
        fitnesses = [evaluate_fitness(populations[i,j], SimpleFitnessFunction()) for i in 1:K, j in 1:N]
        prev_predictions = randn(K)

        before_f = [populations[k, j].f for j in 1:N]
        before_e = [populations[k, j].e for j in 1:N]
        before_fitnesses = squeeze(fitnesses[k, :], 1)
        prev_prediction = prev_predictions[k]

        @fact before_f --> before_fitnesses

        select_for_evolvability(populations, fitnesses, k, prev_predictions, 0, 0, 0, K, N)

        for k2 in 1:K
            after_f = [populations[k2, j].f for j in 1:N]
            after_e = [populations[k2, j].e for j in 1:N]
            after_fitnesses = squeeze(fitnesses[k2, :], 1)
            after_prev_prediction = prev_predictions[k2]

            @fact after_f --> before_f
            @fact after_e --> before_e
            @fact after_fitnesses --> before_fitnesses
            @fact after_prev_prediction --> prev_prediction
        end

    end

end

facts("evolvability_point_estimates()") do
    K = 3
    N = 10
    N2 = 10000
    M = 5

    populations = [SimpleIndividual(0, i) for i in 1:K, j in 1:N]
    fitnesses = [evaluate_fitness(populations[i,j], SimpleFitnessFunction()) for i in 1:K, j in 1:N]
    fitness_function = SimpleFitnessFunction()

    index, c, f = evolvability_point_estimates(1, K, M, N, N2, fitnesses, populations, fitness_function, "std", 0, 0)
    @fact index --> 0
    @fact c --> 0
    @fact f --> 0
    index, c, f = evolvability_point_estimates(0, K, M, N, N2, fitnesses, populations, fitness_function, "std", 0, 0)
    @fact index --> 3
    @fact c --> K * N2
    @fact f --> K * N2

    num = 10000
    count = 0
    for i in 1:num
        index, c, f = evolvability_point_estimates(0, K, M, N, 2, fitnesses, populations, fitness_function, "std", 0, 0)
        if index == 3
            count += 1
        end
    end
    frac = count / num
    @fact frac --> greater_than(1/3)
    @fact frac --> less_than(0.95)
    #println(frac)

end

using StatsBase

facts("tournament_selection()") do
    K = 3
    N = 10

    populations = [SimpleIndividual(j, 100*i) for i in 1:K, j in 1:N]
    fitnesses = [evaluate_fitness(populations[i,j], SimpleFitnessFunction()) for i in 1:K, j in 1:N]
    fitness_function = SimpleFitnessFunction()

    old_fitnesses = copy(fitnesses)
    new_pop, new_fit, obs, sizes = tournament_selection(N, 1000, 3, fitnesses, populations, fitness_function, "std")
    @fact size(new_pop) --> (1000,)
    new_fitnesses = [evaluate_fitness(populations[i, j], fitness_function) for i in 1:K, j in 1:N]
    @fact old_fitnesses --> new_fitnesses

    #println(counts(round(Int64, new_fit+0.55)))
    #println(mean(obs))
    #println(obs)
    #println(sizes)

    #=
    results = zeros(1000)
    for i in 1:1000
        new_pop, new_fit, obs, sizes = tournament_selection(N, 10, 3, fitnesses, populations, fitness_function, "std")
        results[i] = mean(obs)
    end
    println(mean(results))
    =#
end

facts("kalman") do
    kalman = init_kalman(3)

    init_xs = copy(kalman.xs)
    init_ps = copy(kalman.ps)

    counts = zeros(Int64, 3)
    for i in 1:100
        index = rand(1:3)
        counts[index] += 1
        filter_predict(kalman, 0.0, 0, "std", index)
    end
    @fact kalman.xs --> init_xs
    @fact kalman.ps --> roughly(init_ps + diagm(counts), 0.01)

    for ML_q in [0.0, 0.1]
        kalman = init_kalman(3)

        init_xs = copy(kalman.xs)
        init_ps = copy(kalman.ps)

        counts = zeros(Int64, 3)
        for i in 1:100
            index = rand(1:3)
            counts[index] += 1
            filter_predict(kalman, ML_q, 1, "std", index)
        end
        @fact kalman.xs --> init_xs
        @fact kalman.ps --> roughly(init_ps + ML_q * diagm(counts), 0.0001)
    end

    kalman = init_kalman(3)

    for i in 1:3
        obs =
    end
end




















