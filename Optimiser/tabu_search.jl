include("convergence.jl")
include("functions.jl")
include("plot.jl")
include("summarise.jl")
include("utilities.jl")

using PyPlot
using StatsBase

# RNG seed for consistent comparisons
srand(567)

function tabu_search(f::Function,
                    x0::Vector{Float64};
                    max_iters=1000,
                    max_f_evals=1000,
                    x_tol=1e-8,
                    f_tol=1e-8,
                    constraints=[],
                    plot=false,
                    logging=false)

  tic()
  dims = length(x0)

  if length(constraints) > 0
    x_range = constraints
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
    stm = [(x, v); stm]
    stm = stm[1:min(end, STM_SIZE)]

    function add_distinct_point(x,v)
      close_to_existing_point = false

      if DISSIMILARITY_FLAG
        for i in 1:length(mtm)
          if norm(mtm[i][1]-x) < distance_threshold
            close_to_existing_point = true
            if norm(mtm[i][1]-x) <= similar_threshold && v < mtm[i][2]
              mtm[i] = (x,v) # New point is close enough to existing point
            end
            break
          end
        end
      end

      if !close_to_existing_point
        mtm = [mtm; (x, v)]
        mtm = sort(mtm, by=pt->pt[2])[1:min(end, MTM_SIZE)]
      end
    end

    # MTM is sorted list of the N best points
    if !((x,v) in mtm) && (length(mtm) < MTM_SIZE || v < mtm[end][2])
      if length(mtm) == 0 || v < mtm[1][2] # Best I ever had
        mtm = [(x, v); mtm]
        mtm = mtm[1:min(end, MTM_SIZE)]
      else
        add_distinct_point(x,v)
      end
    end

    # Long Term Memory records which areas of search space have had attention.
    ltm = [ltm x]
  end

  function within_constraints(x)
    if length(constraints) > 0
      return minimum(constraints[:,1] .<= x .<= constraints[:,2])
    end
    return true
  end

  function allowed(x)
    seen = x in [pt[1] for pt in stm]
    return !seen && within_constraints(x)
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

    # Add Gaussian noise
    x1_bin_width = x1_bins[2] - x1_bins[1]
    x2_bin_width = x2_bins[2] - x2_bins[1]
    while true
      x1 = x1_bins[x1_index] + x1_bin_width*randn()
      x2 = x2_bins[x2_index] + x2_bin_width*randn()
      if x_range[1,1]<= x1<=x_range[1,2] && x_range[2,1]<=x2<=x_range[2,2]
        return [x1, x2]
      end
    end
  end

  const STM_SIZE = 7
  const MTM_SIZE = 4
  const K_INTENSIFICATION = 10
  const K_DIVERSIFICATION = 15
  const K_STEP_SIZE_REDUCTION = 25
  const STEP_SIZE_MULTIPLIER = 0.1
  const INITIAL_STEP_SIZE = 1.0
  const DISSIMILARITY_FLAG = false
  const distance_threshold = 0.05 # Distance below which points are "close"
  const similar_threshold = 0.004 # Points are similar, keep best.

  stm = Vector[] # Short Term Memory (records last N locations)
  mtm = Vector[] # Medium Term Memory (records the N best solutions)
  ltm = Array(Float64,dims,0) # Long Term Memory records all x
  vals_log = Array(Float64,MTM_SIZE,0)
  f_evals_log = []

  params = [STM_SIZE, MTM_SIZE, K_INTENSIFICATION, K_DIVERSIFICATION,
    K_STEP_SIZE_REDUCTION, STEP_SIZE_MULTIPLIER, INITIAL_STEP_SIZE]

  # Uniform step_size in each direction
  step_size = ones(dims)*INITIAL_STEP_SIZE
  x_base = x0
  v_base = f(x_base)
  f_evals = 1
  update_memory(x_base, v_base)
  iter = 0
  counter = 0

  converged_dict = create_converged_dict(x_tol=x_tol, f_tol=f_tol)

  while true
    # Start of each loop with a new x_base point, and accurate v_base
    iter += 1
    v_base = f(x_base)
    f_evals += 1
    current_best_v = mtm[1][2]

    if plot
      fig, ax2 = plot_tabu(f, x_range, stm, mtm, ltm, iter; name="tabu")
      if iter%10 == 0
        savefig(
          @sprintf "figs/tabu-%s-%s-%04d.pdf" symbol(f) join(params, "-") iter)
      end
    end
    if logging
      mtm_vals = [pt[2] for pt in mtm]
      vals_log = [vals_log pad(mtm_vals, MTM_SIZE, NaN)] # Log MTM over time
      f_evals_log = [f_evals_log; f_evals]
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
      update_memory(x_current, v_current)
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
    if counter == K_INTENSIFICATION
      x_current = mean([pt[1] for pt in mtm])
      v_current = f(x_current)
      f_evals += 1
      update_memory(x_current, v_current)
    elseif counter == K_DIVERSIFICATION
      x_current = diversify(ltm)
      v_current = f(x_current)
      f_evals += 1
      update_memory(x_current, v_current)
    elseif counter == K_STEP_SIZE_REDUCTION
      step_size = step_size .* STEP_SIZE_MULTIPLIER
      x_current = mtm[1][1]
      v_current = mtm[1][2]
      update_memory(x_current, v_current)
      counter = 0
    end

    x_base, v_base = copy(x_current), copy(v_current)
    convergence!(converged_dict; x_step=step_size)
    converged_dict["converged"] && break;
    f_evals >= max_f_evals && break;
    iter >= max_iters && break;
    step_size[1] <= x_tol && break;
  end

  return summarise(mtm, f_evals, toq();
                  converged_dict=converged_dict,
                  x_initial=x0,
                  vals_log=vals_log,
                  f_evals_log=f_evals_log,)
end
