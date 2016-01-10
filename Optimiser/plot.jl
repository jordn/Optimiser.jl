using PyPlot

function plot_contour(f, x_range; name="contour", plot_log=false)

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
  if plot_log
    grid = log(grid)
  end

  fig = figure("surfaceplot", figsize=(10,10))
  ax1 = fig[:add_subplot](2,1,1, projection = "3d")

  ax1[:plot_surface](x1grid, x2grid, grid, rstride=2, edgecolors="k",
    cstride=2, cmap=ColorMap("jet_r"),
    alpha=0.8, linewidth=0.25)
  xlabel("x1")
  ylabel("x2")
  plot_log ? zlabel("log f(x)") : zlabel("f(x)")
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


function plot_slice(f, x_range::Vector; name="line", plot_log=false)
  # Plot (interactive) in external window as updating plots doesn't work in Jupyter
  pygui(true); PyPlot.ion();
  n = 200
  x = linspace(x_range[1], x_range[2], n);
  v = zeros(n)
  v = [f(x_i) for x_i in x]

  if plot_log
    v = [log(f(x_i)) for x_i in x]
  else
    v = [f(x_i) for x_i in x]
  end
  # fig = figure("sliceplot", figsize=(10,10))
  fig = figure()
  # ax = fig[:add_axes]()

  # ax[:plot](x, v)
  plot(x,v)
  ax = gca()
  xlabel("x")
  plot_log? ylabel("log f(x)") : ylabel("f(x)")
  title(@sprintf "Plot of %s" symbol(f))
  tight_layout()
  savefig(@sprintf "figs/%s-%s-0.png" name symbol(f))
  return fig, ax
end
