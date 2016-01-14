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
