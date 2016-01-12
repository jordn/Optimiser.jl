include("convergence.jl")

""" Return a consistent data structure summarising the results. """
function summarise(pts, f_evals, elapsed_time="";
                  g_evals="",
                  converged_dict="",
                  log=[],
                  x_initial=[])

  best_pt = pts[length(pts)]

  summary = Dict{ASCIIString, Any}(
    "x" => best_pt[1],
    "min_value" => best_pt[2],
    "gradient" => length(best_pt)>2 ? best_pt[3] : "N/A",
    "elapsed_time" => elapsed_time,
    "function_evals" => f_evals,
    # "gradient_evals" => g_evals,
    "pts" => pts,
    "convergence" => converged_dict,
    "x_initial" => x_initial,
  )
  if length(log) > 0
    summary["log"] = log
  end
  return summary
end
