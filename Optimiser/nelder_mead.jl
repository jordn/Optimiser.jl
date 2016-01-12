include("convergence.jl")
include("functions.jl")
include("plot.jl")
include("summarise.jl")

using PyPlot

# RNG seed for consistent comparisons
srand(567)

# Following the algorithm described in Lagarias et al
# 'CONVERGENCE PROPERTIES OF THE NELDER–MEAD SIMPLEX METHOD IN LOW DIMENSIONS'
# http://people.duke.edu/~hpgavin/ce200/Lagarias-98.pdf
function nelder_mead(func::Function,
                    x0::Vector{Float64};
                    max_iters=1000,
                    max_f_evals=1000,
                    x_tol=1e-8,
                    f_tol=1e-8,
                    constraints=[],
                    plot=false,
                    logging=false)

  tic()

  if length(constraints) > 0
    x_range = constraints
    function f(x)
      for i in length(x)
        if x[i] < constraints[i,1] || constraints[i,2] < x[i]
          return func(x) + 1e10 # Outside of bounds, penalise.
        end
      end
      return func(x)
    end
  else
    f(x) = func(x)
    x1_max = max(1, abs(x0[1])*2.2)
    x2_max = max(1, abs(x0[2])*2.2)
    x_range = [-x1_max x1_max; -x2_max x2_max]
  end

	if plot
    fig, ax1, ax2 = plot_contour(func, x_range; method="nm")
	end

	const c_reflection, c_expansion, c_contraction, c_shrink = 1.0, 2.0, 0.5, 0.5
  hypers = c_reflection, c_expansion, c_contraction, c_shrink

  iteration = 0;
  f_evals = 0;
  converged_dict = create_converged_dict(x_tol=x_tol, f_tol=f_tol)

  n = length(x0)
	x = repmat(x0, 1, n+1)
	val = zeros(n+1)
	for i = 1:n
		x[i,i] += 1
	end

  pts = []
	simplex = []
  log_vals = Array(Float64,n+1,0)
  log_f_evals = []

	for i = 1:n+1
		val[i] = f(x[:,i]); f_evals += 1;
		push!(simplex, (x[:,i], val[i]))
	end

	while !converged_dict["converged"] && iteration <= max_iters && f_evals <= max_f_evals
		iteration += 1

    # Sort from best to worst
		sort!(simplex, by=pt->pt[2])
    push!(pts, simplex[1])

		if plot
			x1 = [pt[1][1] for pt in simplex]
			x2 = [pt[1][2] for pt in simplex]
      v = [pt[2] for pt in simplex]
			x1, x2 = [x1; x1[1]], [x2; x2[1]] # Add vertex to close simplex

      ax2[:plot](x1, x2, "o--")
      if iteration%1 == 0
        savefig(@sprintf "figs/nelder-%s-%s-%04d.pdf" symbol(f) join(hypers, "-") iteration)
      end
		end

    if logging
       log_vals = [log_vals [pt[2] for pt in simplex]]
       log_f_evals = [log_f_evals; f_evals]
    end

		# x_bar, the centroid of the n best vertices
		x_bar = sum(pt->pt[1], simplex[1:n])./n

		# Reflection
		x_reflection = x_bar + c_reflection*(x_bar - simplex[n+1][1])
		f_reflection = f(x_reflection); f_evals += 1;
		if simplex[1][2] <= f_reflection < simplex[n][2]
			simplex[n+1] = (x_reflection, f_reflection)
			continue
		end

		# Expansion
		if f_reflection < simplex[1][2]
			x_expansion = x_bar + c_expansion*(x_reflection - x_bar)
			f_expansion = f(x_expansion); f_evals += 1;
			if f_expansion < f_reflection
				simplex[n+1] = (x_expansion, f_expansion)
			else
				simplex[n+1] = (x_reflection, f_reflection)
			end
			continue
		end

		# Contraction
		if simplex[n][2] <= f_reflection < simplex[n+1][2]

			# Outside contraction
			x_outside_contraction = x_bar + c_contraction*(x_reflection - x_bar)
			f_outside_contraction = f(x_outside_contraction); f_evals += 1;
			if f_outside_contraction <= f_reflection
				simplex[n+1] = (x_outside_contraction, f_outside_contraction)
				continue
			end

			# Inside contraction
			if f_reflection >= simplex[n+1][2]
				x_inside_contraction = x_bar - c_contraction*(x_reflection - x_bar)
				f_inside_contraction = f(x_inside_contraction); f_evals += 1;
				if f_inside_contraction < simplex[n+1][2]
					simplex[n+1] = (x_inside_contraction, f_inside_contraction)
          continue
				end
			end
		end

		# Shrink simplex
		for i = 2:n+1
			x_i = simplex[1][1] + c_shrink*(simplex[i][1] - simplex[1][1])
			f_i = f(x_i) ; f_evals += 1;
			simplex[i] = (x_i, f_i)
		end

    x = [pt[1] for pt in simplex]
    val = [pt[2] for pt in simplex]
    x_dist = norm(x[end] - x[1])
    val_dist = norm(val[end] - val[1])
    convergence!(converged_dict; x_step=x_dist, f_step=val_dist)

	end

	return summarise(simplex, f_evals, toq();
          converged_dict=converged_dict, x_initial=x0,
          log_vals=log_vals, log_f_evals=log_f_evals)

end
