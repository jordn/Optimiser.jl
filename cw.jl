include("Optimiser/optimise.jl")
include("Optimiser/functions.jl")
# include("Optimiser/nelder_mead.jl")
# include("Optimiser/tabu_search.jl")
include("Optimiser/plot.jl")
srand(567)

function testscalar(f::Function, runs=5)
  tic()
  results = []
  plot_line(f,[-0.05,0.5])

  n = 1000
  x_range = [-1,1]
  for x in linspace(x_range[1], x_range[2], n);
    summary = line_search(multimin, x; plot=false)
    results = [results; summary]
  end
  toc()
  # plot_training(results, f)
  return results
end


# plot_line(multimin, x_range=[-0.1,0.1]; title="x^4*cos(1/x) + 2x^4", name="line")

function testmulti(method::Symbol=:minimise, problem::Symbol=:rosenbrock, runs=1)

  if problem == :rosenbrock
    func = rosenbrock
    start_x = [-0.35,-1.68]
    constraints = [-1.5 1.5; -2 3]
    known_minimum = 0
  end

  if method == :minimise
    method_string = "Gradient"
    optimiser(x) = minimise(func,
                            x;
                            max_f_evals=1000,
                            constraints=constraints,
                            plot=false,
                            logging=true)
  end
  tic()
  results = []

  for i in 1:runs
    x = [rand(linspace(-1.5,1.5)); rand(linspace(-2,3))]
    println(x)
    summary = optimiser(x)
    results = [results; summary]
  end

  tic()
  plot_cumulative_solved(results,
                        known_minimum=known_minimum,
                        method=method_string,
                        problem=problem)
  toc()
  return results
end
