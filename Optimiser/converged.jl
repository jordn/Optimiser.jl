# TODO include reason for convergence
""" Checks for convergence. Supply with key-word args to test various
 convergences. kwargs: x_step, f_step, grad"""
function converged(kwargs...;
                  x_tolerance::Real=1e-8,
                  f_tolerance::Real=1e-8,
                  grad_tolerance::Real=1e-8)

  convergence_tests = []
  for (key,val) in kwargs
    if key == "x_step"
      x_step = val
      x_converged = maximum(abs(x_step)) <= x_tolerance
      push!(convergence_tests, x_converged)
    elseif key == "f_step"
      f_step = val
      f_converged = f_step <= f_tolerance
      push!(convergence_tests, f_converged)
    elseif key == "grad"
      grad = val
      grad_converged = maximum(abs(grad)) <= grad_tolerance
      push!(convergence_tests, grad_converged)
    end
  end
  converged = maximum([convergence_tests; false])
  return converged

end
