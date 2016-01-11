using PyPlot

function plot_contour(f, x_range; name="contour")

  ############
  ##  Plot  ##
  ############

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

  fig = figure("surfaceplot", figsize=(10,10))
  ax1 = fig[:add_subplot](2,1,1, projection = "3d")

  ax1[:plot_surface](x1grid, x2grid, grid, rstride=2, edgecolors="k",
    cstride=2, cmap=ColorMap("jet_r"),
    alpha=0.8, linewidth=0.25)
  xlabel("x1")
  ylabel("x2")
  zlabel("f(x)")
  title(@sprintf "Surface plot of %s" symbol(f))

  subplot(212)
  ax2 = fig[:add_subplot](2,1,2)
  cp = ax2[:contour](x1grid, x2grid, grid, n, linewidth=2.0,
   cmap=ColorMap("jet_r"),)
  xlabel("x1")
  ylabel("x2")
  title(@sprintf "Contour plot of %s" symbol(f))
  tight_layout()

  savefig(@sprintf "figs/%s-%s-0.png" name symbol(f))
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
  savefig(@sprintf "figs/%s-%s-0.png" name symbol(f))
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
  savefig(@sprintf "figs/%s-0.png" name)
  return fig, axarr
end
