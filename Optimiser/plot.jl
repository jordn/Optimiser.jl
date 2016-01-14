using PyPlot
include("utilities.jl")
include("functions.jl")
srand(567)

function plot_contour(f, x_range;
                      name="contour",
                      method="",
                      problem="")

  # Plot (interactive) in external window as updating plots doesn't work in Jupyter
  close("all"); pygui(true); PyPlot.ion();
  n = 200
  x1 = linspace(x_range[1,1], x_range[1,2], n);
  x2 = linspace(x_range[2,1], x_range[2,2], n);
  grid = zeros(length(x2),length(x1))

  x1grid = repmat(x1', length(x2), 1)
  x2grid = repmat(x2, 1, length(x1))

  for i in 1:length(x2) #row (x2[i])
    for j in 1:length(x1) #col (x1[j])
      grid[i:i,j:j] = f([x1[j],x2[i]])
    end
  end

  fig1 = figure("surfaceplot")
  # ax1 = fig1[:add_axes](projection="3d")
  plot_surface(x1grid, x2grid, grid, rstride=5, edgecolors="k",
    cstride=5, cmap=ColorMap("YlGnBu_r"),
    alpha=1.0, linewidth=0.05)
  tight_layout()
  ax1 = gca()
  ax1[:contour](x1grid, x2grid, grid, 30, zdir="z", offset=-7,
    linewidth=1.0, alpha=0.6, cmap=ColorMap("YlGnBu_r"),)
  xlabel("x_1")
  ylabel("x_2")
  zlabel("f(x)")
  title(@sprintf "Surface plot of %s" symbol(f))
  savefig(@sprintf "figs/surface-%s.pdf" symbol(f))

  fig2 = figure("contourplot")
  # ax2 = fig2[:add_axes]()
  cp = contour(x1grid, x2grid, grid, 100, linewidth=2.0, cmap=ColorMap("YlGnBu_r"),)
  ax2 = gca();
  hold(true;)
  xlabel("x_1")
  ylabel("x_2")
  title(@sprintf "Contour plot of %s" problem)
  tight_layout()
  savefig(@sprintf "figs/%s-%s-0.pdf" name symbol(f))
  return fig2, ax1, ax2
end


function plot_tabu(f, x_range, stm, mtm, ltm, iteration;
                      name="tabu",
                      method="",
                      problem="")

  # Plot (interactive) in external window as updating plots doesn't work in Jupyter
  n = 200
  x1 = linspace(x_range[1,1], x_range[1,2], n);
  x2 = linspace(x_range[2,1], x_range[2,2], n);
  grid = zeros(length(x2),length(x1))

  x1grid = repmat(x1', length(x2), 1)
  x2grid = repmat(x2, 1, length(x1))

  for i in 1:length(x2) #row (x2[i])
    for j in 1:length(x1) #col (x1[j])
      grid[i:i,j:j] = f([x1[j],x2[i]])
    end
  end

  fig2 = figure("contourplot")
  cla();
  # ax2 = fig2[:add_axes]()
  cp = contour(x1grid, x2grid, grid, 25, linewidth=2.0, cmap=ColorMap("YlGnBu_r"),)
  ax2 = gca();
  hold(true;)
  xlabel("x_1")
  ylabel("x_2")
  title(@sprintf "Contour plot of %s" symbol(f))
  tight_layout()


  ax2[:plot](ltm[1,:], ltm[2,:], "x", color=[0.1,0.1,0.1], markersize=13)
  for pt in stm
    ax2[:plot](pt[1][1], pt[1][2], "o", color="k", markersize=11)
  end
  for pt in mtm
    ax2[:plot](pt[1][1], pt[1][2], "o", color="r", markersize=11)
  end
  return fig2, ax2
end


function plot_line(f::Function, x_range::Vector; title="", name="line")
  # Plot (interactive) in external window as updating plots doesn't work in Jupyter
  close("lineplot"); pygui(true); PyPlot.ion();
  n = 2000
  x = linspace(x_range[1], x_range[2], n);
  v = zeros(n)
  v = [f(x_i) for x_i in x]

  fig = figure("lineplot")
  # fig = figure()
  # ax = fig[:add_axes]()

  # ax[:plot](x, v)
  plot(x,v, linewidth=2.5, color=[0, 0.4470, 0.7410])
  ax = gca()
  xlabel("x", fontsize=15)
  ylabel("f(x)", fontsize=15)
  # title(@sprintf "Contour plot of %s" symbol(f))
  # grid(true)
  xlim(x_range)
  tight_layout()
  savefig(@sprintf "figs/%s-%s-%f.pdf" name symbol(f) x_range[2])
  return fig, ax
end

function plot_training(results; name="training")
  function add_point(summary)

    pts = summary["pts"]
    first_grad_index = find(pt->pt[3]!=0, pts)[] #First point post bracketting
    pts = pts[first_grad_index:end]

    k = length(pts)
    println(k)
    println(length(summary["f_evals_log"]))
    # x_log = summary["x_log"]
    vals_log = summary["vals_log"]
    grad_log = summary["grad_log"]
    f_evals_log = summary["f_evals_log"]
    grad = [norm(grad_log[:,i]) for i in 1:size(grad_log,2)]
    dims = length(pts[1][1])
    x = Array(Float64,dims,0)
    val = []
    # grad = Array(Float64,dims,0)
    #
    # for pt in pts
    #   x = [x pt[1]]
    #   val = push!(val, pt[2])
    #   grad = [grad pt[3]]
    # end
    # x_steps = [norm(x[i] - x[i-1]) for i in 2:k]

    # axarr[3][:plot](2:k, x_steps, "x-", linewidth=2.0, color=colors[i])
    axarr[1][:plot](f_evals_log, vals_log, "-", linewidth=2.0, color=colors[col_ind+=1])
    if minimum(vals_log) > 0
      axarr[1][:set_yscale]("log")
      axarr[1][:set_ylabel]("f(x) [log scale]")
    else
      axarr[1][:set_ylabel]("f(x)")
    end
    axarr[2][:plot](f_evals_log, grad, "-", linewidth=2.0, color=colors[col_ind])
  end

  runs = length(results)
  summary = results[rand(1:runs)]
  pts = summary["pts"]
  # first_grad_index = find(pt->pt[3]!=0, pts)[] #First point post bracketting
  # pts = pts[1:end]
  method = summary["method"]
  # k = length(pts)
  dims = length(pts[1][1])
  x = Array(Float64,dims,0)
  val = []
  grad = Array(Float64,dims,0)

  vals_log = summary["vals_log"]
  f_evals_log = summary["f_evals_log"]

  if haskey(summary, "grad_log")
    grad_log = summary["grad_log"]
    grad = [norm(grad_log[:,i]) for i in 1:size(grad_log,2)]
  end


  # # for pt in pts
  # #   x = [x pt[1]]
  # #   val = push!(val, pt[2])
  # #   grad = [grad pt[3]]
  # # end
  # x_steps = [norm(x[i] - x[i-1]) for i in 2:k]
  # grad = [norm(grad[i]) for i in 1:k]
  if haskey(summary, "grad_log")
    fig, axarr = plt[:subplots](2, sharex=true)
  else
    fig, axarr = plt[:subplots](1, sharex=true)
  end
  col_ind = 0
  # axarr[3][:plot](2:k, x_steps, "-", linewidth=2.0, color=(0.4,0.4,0.4))
  # axarr[3][:set_ylabel]("\Delta x")

  axarr[1][:plot](f_evals_log, vals_log, "-", linewidth=2.0, color=colors[col_ind+=1])
  if minimum(vals_log) > 0
    axarr[1][:set_yscale]("log", fontsize=13)
    axarr[1][:set_ylabel]("f(x) [log scale]", fontsize=13)
  else
    axarr[1][:set_ylabel]("f(x)")
  end

  if haskey(summary, "grad_log")
    axarr[2][:plot](f_evals_log, grad, "-", linewidth=2.0, color=colors[col_ind])
    axarr[2][:set_ylabel]("gradient [log scale]", fontsize=13)
    axarr[2][:set_yscale]("log")
  end

  xlabel("Function evaluations", fontsize=15)
  axarr[1][:set_title](@sprintf "%s" method)
  tight_layout()
  hold("on")
  for i = 1:3
    summary = results[rand(1:runs)]
    add_point(summary)
  end
  savefig(@sprintf "figs/%s-%s-0.pdf" method name)
  return fig, axarr
end

function plot_cumulative_solved(summaries;
                                name="cumulative",
                                method="",
                                problem="",
                                known_minimum=NaN,
                                f_tol=1e-8)
  close("cumulative")
  runs = length(summaries)
  fig = figure("cumulative")
  max_f_evals = 1000
  range = 1:max_f_evals # iterations we care about
  tally_solved = zeros(max_f_evals)
  tally_close = zeros(max_f_evals)
  loose_f_tol = 1;

  for s in summaries
    # best_val = s["vals_log"][1:min(end,max_f_evals)]
    best_val = s["vals_log"][1:min(end,max_f_evals)]
    f_evals = s["f_evals_log"][1:min(end,max_f_evals)]
    index_to_solve = findfirst(val -> val<=known_minimum+f_tol, best_val)
    index_to_close = findfirst(val -> val<=known_minimum+loose_f_tol, best_val)
    println(index_to_close)
    if index_to_solve != 0
      f_evals_to_solve = f_evals[index_to_solve]
      max_f_evals = maximum([max_f_evals, f_evals[end]])
      tally_solved[f_evals_to_solve] += 1
    end
    if index_to_close != 0
      f_evals_to_close = f_evals[index_to_close]
      tally_close[f_evals_to_close] += 1
    end
  end

  percent_solved = cumsum(tally_solved)/runs
  plot(percent_solved, color=colors[1], linewidth=4.0, alpha=0.5)
  ax = gca()

  percent_close = cumsum(tally_close)/runs
  ax[:plot](percent_close, color=colors[2])
  println(percent_close)

  ax[:fill_between](range-1, percent_close, facecolor=colors[2], alpha=0.2)
  ax[:fill_between](range-1, percent_solved, facecolor=colors[1], alpha=0.3)
  legend([f_tol, loose_f_tol])

  xlabel("Function Evaluations", fontsize=15)
  # xlim(range[1], max_f_evals)
  ylim(0,1)
  # xscale("log")
  ylabel("Global minima found (% of runs)", fontsize=15)
  # legend(["arse", "tits"])
  title((@sprintf "Time to find minima for %s (%i runs)" method runs),fontsize=16)
  tight_layout()
  savefig(@sprintf "figs/%s-%s-%s-%i.pdf" name method problem runs)
  return percent_close
  return fig, ax
end


function plot_gradient(results, title_string="")

  runs = length(results)

  max_width = 0
  for i in 1:runs
    max_width = maximum([max_width, length(results[i]["pts"])])
  end

  # plot(f_evals_log, grad, "-", linewidth=2.0, color=colors[col_ind])
  # axarr[2][:set_ylabel]("gradient [log scale]", fontsize=13)
  # axarr[2][:set_yscale]("log")

  f_runs = reshape(pad([], runs*max_width, NaN), runs, max_width)
  g_runs = reshape(pad([], runs*max_width, NaN), runs, max_width)

  for i in 1:runs
    result = results[i]
    pts = result["pts"]
    for j = 1:length(pts)
      f_runs[i,j] = pts[j][2]
      g_runs[i,j] = norm(pts[j][3])
    end
  end
  close("all")

  for i in 1:runs
    plot(g_runs[i,:]',
      linewidth=2.0, color=colors[i%length(colors)+1], hold=true)
  end
  # plot(g_runs', hold=true)
  ax = gca();
  ax[:set_xlabel]("function evaluations", fontsize=15)
  ax[:set_ylabel]("gradient [log scale]", fontsize=15)
  ax[:set_yscale]("log")
  tight_layout()
  savefig(@sprintf "figs/%s.pdf" lowercase(replace(title_string, " ", "-")))
  return ax
end


function plot_startpoints(results, f::Function; name="training")
  runs = length(results)

  x_runs = []
  x0_runs = []
  f_runs = []
  g_runs = []

  global start_end = Array(Float64, runs, 2)
  global start_end = Array(Float64, runs, 2)
  for i in 1:1
    s = results[i]
    pts = s["pts"]
    x0 = s["x_initial"]
    x = s["x"]
    min_value = s["min_value"]
    gradient = s["gradient"]
    convergence = s["convergence"]
    n = length(pts)
    x_iter = zeros(n)
    f_iter = zeros(n)
    g_iter = zeros(n)
    println(x0)
    println(x)
    start_end[i,:] = [x0 x]
    x_runs = push!(x_runs, x)
    x0_runs = push!(x0_runs, x0)
    f_runs = push!(f_runs, min_value)
    g_runs = push!(g_runs, gradient)

  end
  # close("all")
  # close("minima")
  # plot(x0_runs, x_runs, "o", color=[0.8500,0.3250,0.0980])
  # title("Minima found depending on starting condition")
  # xlabel("Starting point x0", fontsize=15)
  # ylabel("Minima found 'x*''", fontsize=15)
  # savefig(@sprintf "figs/minima-multimin-x.pdf")

  #
  # xrange = [-0.02,0.02]
  # fig, ax = plot_line(f, xrange)
  # println(x_runs)
  # println(f_runs)
  # ax[:plot](x_runs, f_runs, "o", color=[0.8500,0.3250,0.0980], markersize=16)
  # ylim(0,1e-7)
  # title("Minima found for 1000 runs at various start positions")
  # xlabel("x", fontsize=15)
  # ylabel("f(x)", fontsize=15)
  # savefig(@sprintf "figs/minima-multimin-plot-found.pdf")

  return x_runs, x0_runs, f_runs, g_runs

end
