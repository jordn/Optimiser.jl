include("convergence.jl")

""" Return a consistent data structure summarising the results. """
function summarise(pts, f_evals, elapsed_time="";
                  method="",
                  g_evals="",
                  converged_dict="",
                  x_log=[],
                  vals_log=[],
                  grad_log=[],
                  f_evals_log=[],
                  x_initial=[])

  best_pt = pts[end]
  summary = Dict{ASCIIString, Any}(
    "x" => best_pt[1],
    "min_value" => best_pt[2],
    "gradient" => length(best_pt)>2 ? norm(best_pt[3]) : "N/A",
    "elapsed_time" => elapsed_time,
    # "iterations" => iterations,
    "function_evals" => f_evals,
    # "gradient_evals" => g_evals,
    "pts" => pts,
    "method" => method,
    "convergence" => converged_dict,
    "x_initial" => x_initial,
  )
  if length(x_log) > 0
    summary["x_log"] = x_log
  end
  if length(vals_log) > 0
    summary["vals_log"] = vals_log
  end
  if length(grad_log) > 0
    summary["grad_log"] = grad_log
  end
  if length(f_evals_log) > 0
    summary["f_evals_log"] = f_evals_log
  end
  return summary
end
