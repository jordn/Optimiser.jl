include("optimise.jl")
using PyPlot

# Following the algorithm described in Lagarias et al
# 'CONVERGENCE PROPERTIES OF THE NELDER–MEAD SIMPLEX METHOD IN LOW DIMENSIONS'
# http://people.duke.edu/~hpgavin/ce200/Lagarias-98.pdf
function nelder_mead(f::Function, x0, max_iters=500, max_f_evals=1000,
  x_tolerance=1e-6; plot=false, plot_log=false)

  # RNG seed for consistent comparisons
  srand(567)


	if plot
		close("all");
		# Plot in external window as updating plots doesn't work in Jupyter
		pygui(true)
		# tell PyPlot that the plot is interactive
		PyPlot.ion()

		x1scale = max(1, abs(x0[1])*2.2)
		x2scale = max(1, abs(x0[2])*2.2)
		x1 = linspace(-x1scale,x1scale)';
		x2 = linspace(-x2scale,x2scale);
		contour_plot = contour(x1, x2, plot_log? log(f(x1, x2)): f(x1, x2), 400, hold=true)
		ax = gca()
		xlim(-x1scale/1.5, x1scale/1.5)
		ylim(-x2scale/1.5, x2scale/1.5)
		grid("on")
	end

	const c_reflection, c_expansion, c_contraction, c_shrink = 1.0, 2.0, 0.5, 0.5
	n = length(x0)
	x = repmat(x0, 1, n+1)
	f_eval = zeros(n+1)
	for i = 1:n
		x[i,i] += 1
	end

	pts = []
	for i = 1:n+1
		f_eval[i] = f(x[:,i])
		push!(pts, (x[:,i], f_eval[i]))
	end

	function converged(pts)
		x = [pt[1] for pt in pts]
		return norm(x[end] - x[1]) <= x_tolerance
	end

	iterations = 0
  f_evals

	while !converged(pts) && iterations < max_iters
		iterations += 1

		if plot
			x = [pt[1] for pt in pts]
			x = [pt[1] for pt in pts]
			x1 = [pt[1] for pt in x]
			x2 = [pt[2] for pt in x]
			x1, x2 = [x1; x1[1]], [x2; x2[1]] # Add vertex for simplex

			# 2D only for now
			simplex = ax[:plot](x1, x2, "o--")
			if iterations == 10
				ax[:relim]()
				autoscale(tight=false)
			end
			sleep(0.4)

		end

		# Sort from best to worst
		sort!(pts, by=pt->pt[2])

		# x_bar, the centroid of the n best vertices
		x_bar = sum(pt->pt[1], pts[1:n])./n

		# Reflection
		x_reflection = x_bar + c_reflection*(x_bar - pts[n+1][1])
		f_reflection = f(x_reflection)
		if pts[1][2] <= f_reflection < pts[n][2]
			pts[n+1] = (x_reflection, f_reflection)
			continue
		end

		# Expansion
		if f_reflection < pts[1][2]
			x_expansion = x_bar + c_expansion*(x_reflection - x_bar)
			f_expansion = f(x_expansion)
			if f_expansion < f_reflection
				pts[n+1] = (x_expansion, f_expansion)
			else
				pts[n+1] = (x_reflection, f_reflection)
			end
			continue
		end

		# Contraction
		if pts[n][2] <= f_reflection < pts[n+1][2]
			# Outside contraction
			x_outside_contraction = x_bar + c_contraction*(x_reflection - x_bar)
			f_outside_contraction = f(x_outside_contraction)
			if f_outside_contraction <= f_reflection
				pts[n+1] = (x_outside_contraction, f_outside_contraction)
				continue
			end

			# Inside contraction
			if f_reflection >= pts[n+1][2]
				x_inside_contraction = x_bar - c_contraction*(x_reflection - x_bar)
				f_inside_contraction = f(x_inside_contraction)
				if f_inside_contraction < pts[n+1][2]
					pts[n+1] = (x_inside_contraction, f_inside_contraction)
				end
			end
		end

		# Shrink
		for i = 2:n+1
			x_i = pts[1][1] + c_shrink*(pts[i][1] - pts[1][1])
			f_i = f(x_i)
			pts[i] = (x_i, f_i)
		end
	end

	return pts

end

# Takes in two 1D arrays and creates a 2D grid
rosenbrock(x,y) = (1 .- x).^2 .+ 100*(y .- x.^2).^2
# Takes a column vector, or a matrix where each column is an input
rosenbrock{T<:Number}(X::Array{T,2}) = rosenbrock(X[1,:], X[2,:])
rosenbrock{T<:Number}(x::Array{T,1}) = rosenbrock(x[1], x[2])[]
x0 = [0, -10];
# pts = nelder_mead(rosenbrock, x0)
