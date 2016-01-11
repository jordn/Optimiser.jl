""" Checks for convergence. Supply with key-word args to test various
 convergences. kwargs: x_step, f_step, grad"""
function convergence!(converged_dict; x_step=Inf, f_step=Inf, grad=Inf)

  converged_dict["x_converged"] = maximum(abs(x_step)) <= converged_dict["x_tol"]
  converged_dict["f_converged"] = f_step <= converged_dict["f_tol"]
  converged_dict["grad_converged"] = maximum(abs(grad)) <= converged_dict["grad_tol"]
  converged_dict["converged"] = maximum([
    converged_dict["x_converged"],
    converged_dict["f_converged"],
    converged_dict["grad_converged"]
  ])
  return converged_dict
end

function create_converged_dict(;
  x_step=Inf, f_step=Inf, grad=Inf,
  x_tol::Real=1e-8, f_tol::Real=1e-8, grad_tol::Real=1e-8)

  converged_dict = Dict(
    "x_converged"=>false,
    "f_converged"=>false,
    "grad_converged"=>false,
    "converged"=>false,
    "x_tol"=>x_tol,
    "f_tol"=>f_tol,
    "grad_tol"=>grad_tol,
  )
  return convergence!(converged_dict)
end
