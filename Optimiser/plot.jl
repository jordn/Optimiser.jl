using PyPlot
include("utilities.jl")


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
  plot_surface(x1grid, x2grid, grid, rstride=4, edgecolors="k",
    cstride=4, cmap=ColorMap("YlGnBu_r"),
    alpha=1, linewidth=0.05)
  tight_layout()
  ax1 = gca()
  ax1[:contour](x1grid, x2grid, grid, 25, zdir="z", offset=-1, linewidth=1.0, cmap=ColorMap("YlGnBu_r"),)
  xlabel("x_1")
  ylabel("x_2")
  zlabel("f(x)")
  title(@sprintf "Surface plot of %s" symbol(f))
  savefig(@sprintf "figs/%s-%s-0.pdf" name symbol(f))

  fig2 = figure("contourplot")
  # ax2 = fig2[:add_axes]()
  cp = contour(x1grid, x2grid, grid, 25, linewidth=2.0, cmap=ColorMap("YlGnBu_r"),)
  ax2 = gca();
  hold(true;)
  xlabel("x_1")
  ylabel("x_2")
  title(@sprintf "Contour plot of %s" symbol(f))
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

  for i in 1:size(ltm,2)
    ax2[:plot](ltm[1,i], ltm[2,i], "x", color=[0.1,0.1,0.1], markersize=13)
  end

  for pt in stm
    ax2[:plot](pt[1][1], pt[1][2], "o", color="k", markersize=11)
  end
  for pt in mtm
    ax2[:plot](pt[1][1], pt[1][2], "o", color="r", markersize=11)
  end
  return fig2, ax2
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
  most_iterations = 0
  for s in summaries
    mtm = s["log"]
    most_iterations = maximum([most_iterations, size(mtm,2)])
    plot(1:size(mtm,2), mtm[2:end,:]', linewidth=1.0, color=(0.678, 0.675, 0.678), alpha=0.4)
  end
  # Plot best in red, on top of the others.
  for s in summaries
    best_vals = vec(s["log"][1,:])
    plot(1:length(best_vals), best_vals, linewidth=1.0, color=(1, 0.4, 0.4), alpha=0.6)
  end
  range = 1:most_iterations # iterations we care about

  if !isnan(known_minimum)
    plot(range, repmat([known_minimum], most_iterations))
    yticks([yticks()[1]; known_minimum], [yticks()[2], "x*"])
    ylim(-1.25,1)
  end

  xlabel("iteration")
  xlim(range[1], range[end])
  # xscale("log")
  ylabel("minimum f(x) found")

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
    iters_to_close = findfirst(val -> val<=known_minimum+0.001, best_val)
    if iters_to_solve != 0
      tally_solved[iters_to_solve] += 1
    end
    if iters_to_close != 0
      tally_close[iters_to_close] += 1
    end

  end

  percent_solved = cumsum(tally_solved)/runs
  plot(percent_solved, color=vec(colors[1,:]), linewidth=4.0, alpha=0.5)
  ax = gca()

  percent_close = cumsum(tally_close)/runs
  ax[:plot](percent_close, color=vec(colors[2,:]))

  ax[:fill_between](range-1, percent_close, facecolor=vec(colors[2,:]), alpha=0.2)
  ax[:fill_between](range-1, percent_solved, facecolor=vec(colors[1,:]), alpha=0.3)
  legend(["< 1e-8","< 0.001"])

  xlabel("Iteration", fontsize=15)
  xlim(range[1], most_iterations)
  ylim(0,1)
  # xscale("log")
  ylabel("Global minima found (% of runs)", fontsize=15)
  # legend(["arse", "tits"])
  title((@sprintf "Time to find minima for %s (%i runs)" method runs),fontsize=16)
  tight_layout()
  savefig(@sprintf "figs/%s-%s-%s-%i.pdf" name method problem runs)
  return fig, ax
end
