include("Optimiser/optimise.jl")
include("Optimiser/functions.jl")
# include("Optimiser/nelder_mead.jl")
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
  elseif method == :nelder_mead
    method_string = "Nelder Mead"
    optimiser(x) = nelder_mead(func,
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

  # plot_cumulative_solved(results,
  #                       known_minimum=known_minimum,
  #                       method=method_string,
  #                       problem=problem)

  plot_training(results)
  toc()

  return results
end


function quadratic_test(matrix::Symbol=:A10, method::Symbol=:steepdesc, runs=10)
  include("Optimiser/matrix_functions.jl")

  if matrix == :A10
    A = A10
    title="A10"
  elseif matrix == :A100
    A = A100
    title="A100"
  elseif matrix == :A1000
    A = A1000
    title="A1000"
  elseif matrix == :B10
    A = B10
    title="B10"
  elseif matrix == :B100
    A = B100
    title="B100"
  elseif matrix == :B1000
    A = B1000
    title="B1000"
  end

  b = rand(linspace(-1,1,1000), size(A,1))

  if method == :conjgrad
    join([title, "-cg"])
    optimiser(x) = conjgrad(A, b, x)
  elseif method == :steepdesc
    join([title, "-sd"])
    optimiser(x) = steepdesc(A, b, x)
  elseif method == :goldenconj
    join([title, "-goldconj"])
    f(x::Vector) = (1/2 * x'*A*x - b'*x)[]
    g(x::Vector) = A*x - b
    optimiser(x) = minimise(f, x, g, method="conjugate_gradients", max_f_evals=5000)
  elseif method == :goldensteep
    join([title, "-goldsteep"])
    f(x::Vector) = (1/2 * x'*A*x - b'*x)[]
    g(x::Vector) = A*x - b
    optimiser(x) = minimise(f, x, g, method="steepest_descent", max_f_evals=5000)
  end


  results = []
  max_width = 0

  for i in 1:runs
    x0 = rand(linspace(-1,1,1000), size(A,1))
    result = optimiser(x0);
    results = [results; result]
  end

  return results
end


function method_comparison(matrix::Symbol=:A10)
  include("Optimiser/matrix_functions.jl")
  results = []

  # results = [results; quadratic_test(:A100, :steepdesc, 1)]
  results = [results; quadratic_test(:B100, :steepdesc, 1)]
  # results = [results; quadratic_test(:A100, :goldensteep, 1)]
  results = [results; quadratic_test(:B100, :goldensteep, 1)]
  # results = [results; quadratic_test(:A1000, :steepdesc, 1)]
  results = [results; quadratic_test(:B1000, :steepdesc, 1)]
  # results = [results; quadratic_test(:A1000, :goldensteep, 1)]
  results = [results; quadratic_test(:B1000, :goldensteep, 1)]

  # results = [results; quadratic_test(:A100, :goldenconj, 1)]
  # results = [results; quadratic_test(:B100, :goldenconj, 1)]

  ax = plot_gradient(results, "comp")
  xlim(1,100)
  legend([
  # "A100 Exact",
  "B100 Exact",
  # "A100 Golden Section",
  "B100 Golden Section",
  # "A1000 Exact",
  "B1000 Exact",
  # "A1000 Golden Section",
  "B1000 Golden Section",
  # "A100 Golden Conj",
  # "B100 Golden Conj",
  ], loc=0)

  savefig("figs/100matrix-comparison-inex3.pdf")

end
