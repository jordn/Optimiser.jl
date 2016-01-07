include("optimise.jl")

using PyPlot
using StatsBase

function tabu_search(f::Function, x0::Vector{Float64}, max_iters=500, x_tolerance=1e-3; plot=false)

	if plot
		# Plot (interactive) in external window as updating plots doesn't work in Jupyter
		close(); pygui(true); PyPlot.ion();
		x1 = linspace(-2,2)';
		x2 = linspace(-1,1);
		fig_contour = figure(1)
		plot_contour = contour(x1, x2, log(f(x1, x2)), 400, hold=true)
		ax = gca()
		grid("on")
	end

	function update_memory(x, v)
		# Short Term Memory records last N locations
		if length(stm) == stm_size
			shift!(stm)
		end
		stm = [stm; (x, v)] # Equivalent to push! but handles the unintialised array

		# MTM is sorted list of the N best points
		if (length(mtm) < mtm_size || v < mtm[end][2]) && !((x, v) in mtm)
			mtm = [mtm; (x, v)]
			mtm = sort(mtm, by=pt->pt[2])[1:min(end, mtm_size)]
		end

		# Long Term Memory records which areas of search space have had attention.
		x1_index = indmin(abs(bins_x1 - x[1]))
		x2_index = indmin(abs(bins_x2 - x[2]))
		# println(string("($x1_index, $x2_index) for $x"))
		ltm_grid[x2_index, x1_index] += 1
		ltm = [ltm x]
	end

	function allowed(x)
		#  return x in [pt[1] for pt in stm[max(1,end-stm_size):end]]
		seen = x in [pt[1] for pt in stm[max(1,end-stm_size):end]]
		bitarray = [-2.0,-1.0] .<= x .<= [2.0,1.0]
		return bitarray[1] && bitarray[2] && !seen
	end

	function diversify(ltm)
		# Taking a histogram to characterise where has been searched
		fig2 = figure(2)
		tally, x1_bins, x2_bins = plt[:hist2d](vec(ltm[1,:]), vec(ltm[2,:]),
			range=[-2 2; -1 1])

		# Using the negative tallies as an un-normalised distribution for where to
		# search next
		wv = WeightVec(sum(tally)-vec(-tally))
		index = sample(wv)
		x1_index, x2_index = ind2sub((length(x1_bins), length(x2_bins)), index)
		return [x1_bins[x1_index], x2_bins[x2_index]]
	end


	n = length(x0)

	# Uniform increments in each direction for now. TODO optimise.
	increments = ones(n)*0.2

	x_base = x0
	v_base = f(x_base)
	iterations = 0
	counter = 0

	# Short Term Memory (records last N locations)
	const stm_size = 7
	stm = Vector[]

	# Medium Term Memory (records the N best solutions)
	const mtm_size = 4
	mtm = Vector[]

	# Long Term Memory records which areas of search space have had attention.
	const grid_size = 3
	bins_x1 = -2:(4.0/grid_size):2
	bins_x2 = -1:(2.0/grid_size):1
	ltm_grid = zeros(length(bins_x1), length(bins_x2))
	ltm = Array(Float64,2,0)

	const TRIGGER_INTENSIFICATION = 10 # number of iteraions without MTM changing
	const TRIGGER_DIVERSIFICATION = 15
	const TRIGGER_STEP_SIZE_REDUCTION = 25 # number of iteraions without MTM changing

	converged() = false # TODO test convergence

	update_memory(x_base, v_base)

	while !converged() && iterations < max_iters
		iterations += 1
		println("$iterations $counter")

		v_base = f(x_base)
		current_best_v = mtm[1][2]

		if plot
			fig_contour = figure(1)
			ax[:plot](x_base[1], x_base[2], "o--")
			sleep(0.04)
		end

		# LOCAL SEARCH
		x_tests = repmat(x_base, 1, n^2)
		j = 0
		# Increment and decrement each dimension
		for i = 1:n
			x_tests[i,j+=1] += increments[i]
			x_tests[i,j+=1] -= increments[i]
		end
		v_tests = f(x_tests)

		# Select best direction
		order = sortperm(v_tests)
		for i = 1:length(v_tests)
			if allowed(x_tests[:, order[i]])
				x_current = x_tests[:, order[i]]
				v_current = v_tests[order[i]]
				break
			end
		end

		# PATTERN MOVE
		if isdefined(:v_current) && v_current < v_base
			x_test = x_current + x_current - x_base
			v_test = f(x_test)
			if v_test < v_current
				x_current, v_current = x_test, v_test
			end
		end

		update_memory(x_current, v_current)

		if mtm[1][2] < current_best_v # New best
			counter = 0
		else
			counter += 1
		end

		# SEARCH INTENSIFICATION
		if counter == TRIGGER_INTENSIFICATION
			x_current = mean([pt[1] for pt in mtm])
			println("SEARCH INTENSIFICATION DANCE ༼ つ ◕_◕ ༽つ")
		elseif counter == TRIGGER_DIVERSIFICATION
			println("SEARCH DIVERSIFICATION SHRUG ¯\_(ツ)_/¯")
			# TODO Sample from the search space in an unused
			# Sample from unused space
			# index = indmin(ltm_grid)
			# x1_index, x2_index = ind2sub(size(ltm_grid), index)
			# x_current = [bins_x1[x1_index], bins_x2[x2_index]]
			x_current = diversify(ltm)

			println("SEARCH DIVERSIFICATION MOVED TO ", x_current)
		elseif counter == TRIGGER_STEP_SIZE_REDUCTION
			println("STEP SIZE FUCKING REDUCED CUS WHY NOT (╯°□°）╯︵ ┻━┻")
			const ϕ = 0.5 * (1.0 + sqrt(5.0))
			increments = increments/ϕ
			x_current = mtm[1][1]
			counter = 0
		end
		x_base, v_base = x_current, f(x_current)

	end

	return stm,mtm,ltm
end

# Takes in two 1D arrays and creates a 2D grid_size
rosenbrock(x,y) = (1 .- x).^2 .+ 100*(y .- x.^2).^2
# Takes a matrix where each column is an input, returns a vector
rosenbrock{T<:Number}(X::Array{T,2}) = vec(rosenbrock(X[1,:], X[2,:]))
# Takes a vector, returns a number
rosenbrock{T<:Number}(x::Array{T,1}) = rosenbrock(x[1], x[2])[]
x0 = [0.1, -1];
# pts = tabu_search(rosenbrock, x0)
