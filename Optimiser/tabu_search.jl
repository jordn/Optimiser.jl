include("optimise.jl")

using PyPlot

type Point{T}
  x::T
  y::T
end

function tabu_search(f::Function, x0, max_iters=500, x_tolerance=1e-3; plot=false)

	if plot
		# Plot (interactive) in external window as updating plots doesn't work in Jupyter
		close(); pygui(true); PyPlot.ion();
		x1 = linspace(-2,2)';
		x2 = linspace(-1,1);
		contour_plot = contour(x1, x2, log(f(x1, x2)), 400, hold=true)
		ax = gca()
		grid("on")
	end

	n = length(x0)

	# Uniform increments in each direction for now. TODO optimise.
	increments = ones(n)*0.4

	x_base = x0
	v_base = f(x_base)
	iterations = 0
	counter = 0

	# Short Term Memory (records last N locations)
	const stm_size = 7
	stm = [(x_base, v_base)]

	# Medium Term Memory (records the N best solutions)
	const mtm_size = 4
	mtm = [(x_base, v_base)]
	# TODO, incorporate the 'best' into the MTM
	x_best, v_best = x_base, v_base

	# Long Term Memory (records which areas of search space have had attention)
	# TODO make Bayesian
	const grid_size = 10
	ltm_x1 = -2:(4.0/grid_size):2
	ltm_x2 = -1:(2.0/grid_size):1
	ltm = zeros(length(ltm_x1), length(ltm_x2))

	const TRIGGER_INTENSIFICATION = 10 # number of iteraions without MTM changing
	const TRIGGER_DIVERSIFICATION = 2 #TRIGGER_INTENSIFICATION + 2
	const TRIGGER_STEP_SIZE_REDUCTION = 25 # number of iteraions without MTM changing

	function update_memory(x, v)

		stm = [stm; (x, v)]

		mtm = [mtm, (x, v)]

		x1_index = indmin(abs(ltm_x1 - x[1]))
		x2_index = indmin(abs(ltm_x2 - x[2]))
		println(string("($x1_index, $x2_index) for $x_current"))
		ltm[x2_index, x1_index] += 1
	end

	typeof(stm)
	 function allowed(x)
		#  return x in [pt[1] for pt in stm[max(1,end-stm_size):end]]
		seen = x in [pt[1] for pt in stm[max(1,end-stm_size):end]]
		bitarray = [-2.0,-1.0] .<= x .<= [2.0,1.0]
		return bitarray[1] && bitarray[2] && !seen
	 end

	while iterations < max_iters
		iterations += 1
		counter += 1
		# SEARCH INTENSIFICATION
		if counter == TRIGGER_INTENSIFICATION
			x_base = mean([pt[1] for pt in mtm])
			println("SEARCH INTENSIFICATION DANCE ༼ つ ◕_◕ ༽つ")
		elseif counter == TRIGGER_DIVERSIFICATION
			println("SEARCH DIVERSIFICATION SHRUG ¯\_(ツ)_/¯")
			# TODO Sample from the search space in an unused
			index = indmin(ltm)
			x1_index, x2_index = ind2sub(size(ltm), index)
			x_base = [ltm_x1[x1_index], ltm_x2[x2_index]]
			ltm[x1_index, x2_index] += 1 # TODO, overcounting
			counter = 0 #TODO remove
			println("SEARCH DIVERSIFICATION MOVED TO ", x_base)
		elseif count == TRIGGER_STEP_SIZE_REDUCTION
			println("STEP SIZE FUCKING REDUCED CUS WHY NOT (╯°□°）╯︵ ┻━┻")
			increments = 0.618*increments
			counter = 0
		end
		v_base = f(x_base)

		if plot
			# 2D only for now
			ax[:plot](x_base[1], x_base[2], "o--")
			# if iterations == 4
			# 	ax[:relim]()
			# 	autoscale(tight=false)
			# end
			sleep(0.004)
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
			if allowed(x_tests[:, order[i]])
				x_current = x_tests[:, order[i]]
				v_current = v_tests[order[i]]
				push!(stm, (x_current, v_current))
				# TODO Add methods for MTM to judge whether something should be added, order
				push!(mtm, (x_current, v_current))
				x1_val = x_current[1]
				x2_val = x_current[2]
				x1_index = indmin(abs(ltm_x1-x1_val))
				x2_index = indmin(abs(ltm_x2-x2_val))
				# x1_index2 = max(1,floor(Int64, (x_current[1]+2)/4.0*grid_size))
				# x2_index2 = max(1,floor(Int64, (x_current[2]+1)/2.0*grid_size))
				println(string("($x1_index, $x2_index) for $x_current"))
				ltm[x2_index, x1_index] += 1
				if v_current < v_best
					x_best, v_best = x_current, v_current
					counter = 0
				end
				break
			end
		end


		# PATTERN MOVE
		if isdefined(:v_current) && v_current < v_base
			x_test = x_current + x_current - x_base
			v_test = f(x_test)
			if v_test < v_current
				x_current, v_current = x_test, v_test
				push!(stm, (x_current, v_current))
				# TODO Add methods for MTM to judge whether something should be added, order
				push!(mtm, (x_current, v_current))
				x1_index = indmin(abs(ltm_x1-x1_val))
				x2_index = indmin(abs(ltm_x2-x2_val))
				# x1_index2 = max(1,floor(Int64, (x_current[1]+2)/4.0*grid_size))
				# x2_index2 = max(1,floor(Int64, (x_current[2]+1)/2.0*grid_size))
				println(string("($x1_index, $x2_index) for $x_current"))
				ltm[x2_index, x1_index] += 1
				if v_current < v_best
					x_best, v_best = x_current, v_current
					counter = 0
				end
			end
		end

		x_base, v_base = x_current, v_current
		# println(ltm)
	end
	println(x_best, v_best)

	return stm,mtm,ltm

end

# Takes in two 1D arrays and creates a 2D grid_size
rosenbrock(x,y) = (1 .- x).^2 .+ 100*(y .- x.^2).^2
# Takes a matrix where each column is an input, returns a vector
rosenbrock{T<:Number}(X::Array{T,2}) = vec(rosenbrock(X[1,:], X[2,:]))
# Takes a vector, returns a number
rosenbrock{T<:Number}(x::Array{T,1}) = rosenbrock(x[1], x[2])[]
x0 = [0, -10];
# pts = tabu_search(rosenbrock, x0)
