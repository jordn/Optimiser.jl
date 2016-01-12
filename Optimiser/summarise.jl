include("convergence.jl")

""" Return a consistent data structure summarising the results. """
function summarise(pts, f_evals, elapsed_time="";
                  g_evals="",
                  converged_dict="",
                  log_vals=[],
                  log_f_evals=[],
                  x_initial=[])

  best_pt = pts[length(pts)]

  summary = Dict{ASCIIString, Any}(
    "x" => best_pt[1],
    "min_value" => best_pt[2],
    "gradient" => length(best_pt)>2 ? best_pt[3] : "N/A",
    "elapsed_time" => elapsed_time,
    # "iterations" => iterations,
    "function_evals" => f_evals,
    # "gradient_evals" => g_evals,
    "pts" => pts,
    "convergence" => converged_dict,
    "x_initial" => x_initial,
  )
  if length(log_vals) > 0
    summary["log_vals"] = log_vals
  end
  if length(log_f_evals) > 0
    summary["log_f_evals"] = log_f_evals
  end
  return summary
end
