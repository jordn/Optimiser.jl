# include("Optimiser/optimise.jl")
include("Optimiser/nelder_mead.jl")
include("Optimiser/tabu_search.jl")
include("Optimiser/functions.jl")
include("Optimiser/plot.jl")
srand(567) # NB: Include this only once otherwise

function multirun(method::Symbol, problem::Symbol=:camel, runs=4)

  if problem == :camel
    func = camel
    contraints = [-2 2; -1 1]
    known_minimum = -1.031628
  elseif problem == :rosenbrock
    func = rosenbrock
    contraints = [-2 2; -2 2]
  end

  if method == :tabu_search
    method_string = "Tabu Search"
    optimiser(x) = tabu_search(func,
                              x;
                              max_f_evals=1000,
                              contraints=contraints,
                              plot=false,
                              logging=true)

  elseif method == :nelder_mead
    method_string = "Nelder-Mead"
    optimiser(x) = nelder_mead(func,
                              x;
                              max_f_evals=1000,
                              contraints=contraints,
                              plot=false,
                              logging=true)
  end

  tic()
  results = []

  for i in 1:runs
    x = [rand(linspace(-2,2)); rand(linspace(-1,1))]
    summary = optimiser(x)
    results = [results; summary]
  end

  toc()


  if method == :tabu_search
    plot_mtms(results, known_minimum=known_minimum)
  end
  
  plot_cumulative_solved(results,
                        known_minimum=known_minimum,
                        method=method_string,
                        problem=problem)

  return results
end
