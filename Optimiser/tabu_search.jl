include("optimise.jl")

using PyPlot

function tabu_search(f::Function, x0, max_iters=500, x_tolerance=1e-3; plot=false)

	if plot
		# Plot (interactive) in external window as updating plots doesn't work in Jupyter
		close(); pygui(true); PyPlot.ion();
		x1scale = max(1, abs(x0[1])*2.2)
		x2scale = max(1, abs(x0[2])*2.2)
		x1 = linspace(-x1scale,x1scale)';
		x2 = linspace(-x2scale,x2scale);
		contour_plot = contour(x1, x2, log(f(x1, x2)), 400, hold=true)
		ax = gca()
		xlim(-x1scale/1.5, x1scale/1.5)
		ylim(-x2scale/1.5, x2scale/1.5)
		grid("on")
	end



	n = length(x0)

	# Uniform increments in each direction for now. TODO optimise.
	increments = ones(n)

	x_base = x0
	v_base = f(x_base)
	iterations = 0

	# Short Term Memory
	stm_size = 7
	stm = [(x_base, v_base)]
	typeof(stm)
	seen(x) = x in [pt[1] for pt in stm[max(1,end-stm_size):end]]

	while iterations < max_iters

		iterations += 1
		println(iterations)

		if plot
			# 2D only for now
			ax[:plot](x_base[1], x_base[2], "o--")
			# if iterations == 4
			# 	ax[:relim]()
			# 	autoscale(tight=false)
			# end
			sleep(0.4)
		end


		x_tests = repmat(x_base, 1, n^2)
		j = 0
		# Increment and decrement each dimension
		for i = 1:n
			x_tests[i,j+=1] += increments[i]
			x_tests[i,j+=1] -= increments[i]
		end
		v_tests = f(x_tests)

		order = sortperm(v_tests)
		for i = 1:length(v_tests)
			if !seen(x_tests[:, order[i]])
				x_current = x_tests[:, order[i]]
				v_current = v_tests[order[i]]
				# println(x_tests, v_tests)
				println((x_current, v_current))
				push!(stm, (x_current, v_current))
				break
			end
		end

		# Pattern move
		if v_current < v_base
			x_test = x_current + x_current - x_base
			v_test = f(x_test)
			if v_test < v_current
				x_current, v_current = x_test, v_test
				push!(stm, (x_current, v_current))
			end
		end

		x_base, v_base = x_current, v_current
		println(x_base, v_base)
	end


end

# Takes in two 1D arrays and creates a 2D grid
rosenbrock(x,y) = (1 .- x).^2 .+ 100*(y .- x.^2).^2
# Takes a matrix where each column is an input, returns a vector
rosenbrock{T<:Number}(X::Array{T,2}) = vec(rosenbrock(X[1,:], X[2,:]))
# Takes a vector, returns a number
rosenbrock{T<:Number}(x::Array{T,1}) = rosenbrock(x[1], x[2])[]
x0 = [0, -10];
# pts = tabu_search(rosenbrock, x0)
