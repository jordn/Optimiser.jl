include("optimise.jl")
include("plot.jl")

using PyPlot
using StatsBase

function tabu_search(f::Function, x0::Vector{Float64}, max_iters=500,
  max_f_evals=1000, x_tolerance=1e-6; contraints=[], plot=false)

  # RNG seed for consistent comparisons
  srand(567)

  if length(contraints) > 0
    x_range = contraints
  else
    x1_max = max(1, abs(x0[1])*2.2)
    x2_max = max(1, abs(x0[2])*2.2)
    x_range = [-x1_max x1_max; -x2_max x2_max]
  end

  if plot
    fig, ax1, ax2 = plot_contour(f, x_range; name="tabu")
  end

  function update_memory(x, v)
    # Short Term Memory records last N locations
    if length(stm) == STM_SIZE
      shift!(stm)
    end
    stm = [stm; (x, v)] # Equivalent to push! but handles the unintialised array

    # MTM is sorted list of the N best points
    if (length(mtm) < MTM_SIZE || v < mtm[end][2]) && !((x, v) in mtm)
      mtm = [mtm; (x, v)]
      mtm = sort(mtm, by=pt->pt[2])[1:min(end, MTM_SIZE)]
    end

    # Long Term Memory records which areas of search space have had attention.
    ltm = [ltm x]
  end

  function within_contraints(x)
    if length(contraints) > 0
      return minimum(contraints[:,1] .<= x .<= contraints[:,2])
    end
    return true
  end

  function allowed(x)
    seen = x in [pt[1] for pt in stm[max(1,end-STM_SIZE):end]]
    return !seen && within_contraints(x)
  end

  function diversify(ltm)
    # Taking a histogram to characterise where has been searched
    const bins = 6
    x1_bins = linspace(x_range[1,1], x_range[1,2], bins);
    x2_bins = linspace(x_range[2,1], x_range[2,2], bins);
    x1_bins, x2_bins, tally = hist2d(ltm', x1_bins, x2_bins)
    # Using the negative tallies as an unnormalised distribution as we wish to
    # favour unexplored regions.
    wv = WeightVec(sum(tally)-vec(-tally))
    index = sample(wv)
    x1_index, x2_index = ind2sub((length(x1_bins), length(x2_bins)), index)
    return [x1_bins[x1_index], x2_bins[x2_index]]
  end

  # Short Term Memory (records last N locations)
  const STM_SIZE = 7
  stm = Vector[]

  # Medium Term Memory (records the N best solutions)
  const MTM_SIZE = 4
  mtm = Vector[]

  # Long Term Memory records which areas of search space have had attention.
  ltm = Array(Float64,2,0)

  const TRIGGER_INTENSIFICATION = 10 # number of iteraions without MTM changing
  const TRIGGER_DIVERSIFICATION = 15
  const TRIGGER_STEP_SIZE_REDUCTION = 25 # number of iteraions without MTM changing
  const STEP_SIZE_MULTIPLIER = 0.5

  # TODO, use hyperparmeters to sat figure save name
  hypers = [STM_SIZE, MTM_SIZE, TRIGGER_INTENSIFICATION, TRIGGER_DIVERSIFICATION,
    TRIGGER_STEP_SIZE_REDUCTION, STEP_SIZE_MULTIPLIER]

  converged() = minimum(step_size .<= x_tolerance)

  dims = length(x0)

  # Uniform step_size in each direction for now. TODO optimise.
  step_size = ones(dims)*0.08
  x_base = x0
  v_base = f(x_base)
  iterations = 0
  counter = 0
  f_evals = 1


  while !converged() && f_evals <= max_f_evals && iterations <= max_iters
    update_memory(x_base, v_base)

    iterations += 1
    println("$iterations $counter $f_evals")

    v_base = f(x_base)
    f_evals += 1
    current_best_v = mtm[1][2]

    if plot
      ax1[:plot]([x_base[1]], [x_base[2]], v_base, "o--")
      ax2[:plot](x_base[1], x_base[2], "o--")
      if iterations%100 == 0
        savefig(@sprintf "figs/tabu-%s-%d.png" symbol(f) iterations)
      end
      # sleep(0.04)
    end

    x_steps = Array(Float64,2,0)

    # LOCAL SEARCH
    # Increment and decrement each dimension
    for i =1:dims
      x_inc, x_dec = copy(x_base), copy(x_base)
      x_inc[i] += step_size[i]
      x_dec[i] -= step_size[i]
      if allowed(x_inc) x_steps = [x_steps x_inc] end
      if allowed(x_dec) x_steps = [x_steps x_dec] end
    end

    if length(x_steps) > 0
      v_steps = [f(vec(x_steps[:,i])) for i in 1:size(x_steps,2)]
      f_evals += length(v_steps)
      # Select best direction
      v_current, index = findmin(v_steps)
      x_current = x_steps[:,index]
      update_memory(x_current, v_current)   # TODO? put this here?

      # Pattern move
      if v_current < v_base
        x_test = x_current + x_current - x_base
        v_test = f(x_test)
        f_evals += 1
        if v_test < v_current
          x_current, v_current = x_test, v_test
          update_memory(x_current, v_current)
        end
      end
    end

    # Check if new best
    if mtm[1][2] < current_best_v
      counter = 0
    else
      counter += 1
    end

    # SEARCH INTENSIFICATION
    if counter == TRIGGER_INTENSIFICATION
      x_current = mean([pt[1] for pt in mtm])
      v_current = f(x_current)
      f_evals += 1
      println("SEARCH INTENSIFICATION DANCE ༼ つ ◕_◕ ༽つ")
    elseif counter == TRIGGER_DIVERSIFICATION
      println("SEARCH DIVERSIFICATION SHRUG ¯\_(ツ)_/¯")
      x_current = diversify(ltm)
      v_current = f(x_current)
      f_evals += 1
      println("SEARCH DIVERSIFICATION MOVED TO ", x_current)
    elseif counter == TRIGGER_STEP_SIZE_REDUCTION
      println("STEP SIZE REDUCED CUS WHY NOT RAAAHH (╯°□°）╯︵ ┻━┻")
      step_size = step_size .* STEP_SIZE_MULTIPLIER
      x_current = mtm[1][1]
      v_current = mtm[1][2]
      counter = 0
    end
    x_base, v_base = x_current, v_current

  end

  return stm,mtm,ltm
end
