""" Checks for convergence. Supply with key-word args to test various
 convergences. kwargs: x_step, f_step, grad"""
function convergence(;x_step=Inf, f_step=Inf, grad=Inf, x_tol::Real=1e-8,
                  f_tol::Real=1e-8, grad_tol::Real=1e-8)

  x_converged = maximum(abs(x_step)) <= x_tol
  f_converged = f_step <= f_tol
  grad_converged = maximum(abs(grad)) <= grad_tol
  converged = maximum([x_converged; f_converged; grad_converged; false])
  return converged, x_converged, f_converged, grad_converged
end
