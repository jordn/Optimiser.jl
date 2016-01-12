using PyPlot


function plot_contour(f, x_range; name="contour")

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

  fig = figure("surfaceplot", figsize=(15,5))
  ax1 = fig[:add_subplot](1,2,1, projection="3d")
  ax1[:plot_surface](x1grid, x2grid, grid, rstride=4, edgecolors="k",
    cstride=4, cmap=ColorMap("jet_r"),
    alpha=1, linewidth=0.05)
  ax1[:contour](x1grid, x2grid, grid, 30, zdir="z", offset=-1, linewidth=1.0, cmap=ColorMap("jet_r"),)
  xlabel("x1")
  ylabel("x2")
  zlabel("f(x)")
  title(@sprintf "Surface plot of %s" symbol(f))

  subplot(122)
  ax2 = fig[:add_subplot](1,2,2)
  cp = ax2[:contour](x1grid, x2grid, grid, n, linewidth=2.0, cmap=ColorMap("jet_r"),)
  xlabel("x1")
  ylabel("x2")
  title(@sprintf "Contour plot of %s" symbol(f))
  tight_layout()

  savefig(@sprintf "figs/%s-%s-0.pdf" name symbol(f))
  return fig, ax1, ax2
end


function plot_line(f, x_range::Vector; name="line")
  # Plot (interactive) in external window as updating plots doesn't work in Jupyter
  close("lineplot"); pygui(true); PyPlot.ion();
  n = 200
  x = linspace(x_range[1], x_range[2], n);
  v = zeros(n)
  v = [f(x_i) for x_i in x]

  fig = figure("lineplot")
  # fig = figure()
  # ax = fig[:add_axes]()

  # ax[:plot](x, v)
  plot(x,v)
  ax = gca()
  xlabel("x")
  ylabel("f(x)")
  title(@sprintf "Plot of %s" symbol(f))
  tight_layout()
  savefig(@sprintf "figs/%s-%s-0.pdf" name symbol(f))
  return fig, ax
end

function plot_training(pts; name="training")
  # Plot (interactive) in external window as updating plots doesn't work in Jupyter
  k = length(pts)
  dims = length(pts[1][1])
  x = Array(Float64,dims,0)
  val = []
  grad = Array(Float64,dims,0)
  for pt in pts
    x = [x pt[1]]
    val = push!(val, pt[2])
    grad = [grad pt[3]]
  end
  x_steps = [norm(x[i] - x[i-1]) for i in 2:k]
  grad = [norm(grad[i]) for i in 1:k]

  # fig = figure("training", figsize=(6,12))
  # fig = figure("training")
  # fig = figure("surfaceplot", figsize=(6,12))

  fig, axarr = plt[:subplots](3, sharex=true)
  # fig[:set_size_inches](6, 12)
  axarr[1][:plot](2:k, x_steps, "x-", linewidth=2.0, color=(0.4,0.4,0.4))
  axarr[1][:set_ylabel]("\Delta x")

  axarr[2][:plot](1:k, val, "x-", linewidth=2.0, color=(0.4,0.4,0.4))
  if minimum(val) > 0
    axarr[2][:set_yscale]("log")
    axarr[2][:set_ylabel]("f(x) [log scale]")
  else
    axarr[2][:set_ylabel]("f(x)")
  end

  axarr[3][:plot](1:k, grad, "x-", linewidth=2.0, color=(0.4,0.4,0.4))
  axarr[3][:set_ylabel]("gradient [log scale]")
  axarr[3][:set_yscale]("log")
  xlabel("iteration")

  axarr[1][:set_title](@sprintf "%s" name)
  tight_layout()
  savefig(@sprintf "figs/%s-0.pdf" name)
  return fig, axarr
end


function plot_mtms(summaries; name="mtm", known_minimum=NaN)
  close("mtm")
  runs = length(summaries)
  fig = figure("mtm")
  MTM_SIZE = size(summaries[1]["log"],1)
  ax = fig[:add_axes](hold=true)
  max_iterations = 500
  most_iterations = 0
  # for s in summaries
  #   mtm = s["log"]
  #   plot(range, mtm[2:end,range]', linewidth=1.0, color=(0.678, 0.675, 0.678), alpha=0.4)
  # end
  # Plot best in red, on top of the others.
  for s in summaries
    best_vals = vec(s["log"][1,:])
    most_iterations = maximum([most_iterations, length(best_vals)])
    plot(1:length(best_vals), best_vals, linewidth=1.0, color=(1, 0.4, 0.4), alpha=0.6)
  end
  range = 1:most_iterations # iterations we care about

  if !isnan(known_minimum)
    plot(range, repmat([known_minimum], most_iterations))
    yticks([yticks()[1]; known_minimum], [yticks()[2], "x*"])
    ylim(-2,1)
  end

  xlabel("iteration")
  xlim(range[1], range[end])
  # xscale("log")
  ylabel("minimum f(x) found")

  # legend(["arse", "tits"])
  title(@sprintf "Best %i values found for %i runs" MTM_SIZE runs)
  tight_layout()
  savefig(@sprintf "figs/%s-0.pdf" name)
  return fig, ax
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

  max_iterations = 500
  most_iterations = 0
  range = 1:max_iterations # iterations we care about
  tally_solved = zeros(max_iterations)
  tally_close = zeros(max_iterations)

  for s in summaries
    best_val = s["log"][1,1:min(end,max_iterations)]
    most_iterations = maximum([most_iterations, length(best_val)])
    iters_to_solve = findfirst(val -> val<=known_minimum+f_tol, best_val)
    iters_to_close = findfirst(val -> val<=known_minimum+f_tol*100000000, best_val)
    if iters_to_solve != 0
      tally_solved[iters_to_solve] += 1
    end
    if iters_to_close != 0
      tally_close[iters_to_close] += 1
    end

  end

  percent_solved = cumsum(tally_solved)/runs
  plot(percent_solved, color="blue", linewidth=4.0, alpha=0.5)
  ax = gca()
  ax[:fill_between](range-1, percent_solved, facecolor="blue", alpha=0.3)

  # percent_close = cumsum(tally_close)/runs
  # ax[:plot](percent_close)
  # ax[:fill_between](range, percent_close, facecolor="red", alpha=0.2)


  xlabel("iteration")
  xlim(range[1], most_iterations)
  ylim(0,1)
  # xscale("log")
  ylabel("Global minima found (Cumulative % of runs)")
  # legend(["arse", "tits"])
  title(@sprintf "Percentage of %s searches finding global minima (%i runs)" method runs)
  tight_layout()
  savefig(@sprintf "figs/%s-%s-%s-%i.pdf" name method problem runs)
  return fig, ax
end
