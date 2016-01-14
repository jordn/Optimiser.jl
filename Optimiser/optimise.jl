# using Debug
include("plot.jl")
include("functions.jl")
# include("matrix_functions.jl")
include("convergence.jl")
include("summarise.jl")

srand(567)
const ϕ = golden
const global disp_progress = true
normalise(x) = x/norm(x)

"""Return a function that returns the jacobian
(2D input, 1D output only)"""
function jacobian(f::Function, δ=1e-8)
  g(x) = [(f(x+[δ; 0]) - f(x-[δ; 0]))./(2δ)
          (f(x .+ [0; δ;]) - f(x .- [0; δ]))./(2δ);]
end

""" Returns a functions which will approximate the gradient using symmetric
finite difference """
function gradient_approximator(f::Function, δ=1e-12; dims=1)
  # TODO keep track of how many times f has beeen evaluated
  if dims == 1
    g(x) = (f(x + δ) - f(x - δ))/2δ
  elseif dims == 2
    g(x) = [(f(x+[δ; 0]) - f(x-[δ; 0]))/(2δ);
            (f(x+[0; δ]) - f(x-[0; δ]))/(2δ)]
  end
end

"Bracket the minimum of the function such that xa < xb < xc and
    f(xa) > f(xb) < f(xc)"
function bracket(f::Function, xb::Real=0; xa::Real=xb-(1-1/ϕ), xc::Real=xb+1/ϕ,
                 max_evals::Int=1000)

  xa, xb, xc = sort([xa, xb, xc])

  fa = f(xa)
  fb = f(xb)
  fc = f(xc)
  evals = 3

  # Sort the points such that fc > fa. Then take geometrically increasing steps
  # in the direction of xa until fa > fb
  if fa > fc
    xa, xc = xc, xa
    fa, fc = fc, fa
  end

  pts = [(xb,fb,0.0)]
  while fb >= fa
    @printf "(%0.2f, %0.2f, %0.2f) = %0.2f, %0.2f, %0.2f\n" xa xb xc fa fb fc
    xb, xc = xa, xb
    fb, fc = fa, fb
    push!(pts, (xb,fb,0.0))
    xa = xa - ϕ*(xc-xb) # big jump
    fa = f(xa); evals += 1;
    while fa == fb && evals <= max_evals
      xa = xa - ϕ*(xc-xb) # big jump
      fa = f(xa); evals += 1;
    end
    if evals > max_evals
      error("Too many evaluations while attempting to bracket function minimum.")
    end
  end

  if xa > xc
    xa, xc = xc, xa
    fa, fc = fc, fa
  end
  return xa, xb, xc, fa, fb, fc, pts, evals
end

"""Given four points return the smallest range that brackets a minimum"""
function rebracket(xa, xb, xc, x_new, fa, fb, fc, f_new)
  pts = [(xa, fa), (xb, fb), (xc, fc), (x_new, f_new)]
  x_order = sortperm(pts, by=pt->pt[1])
  ordered_pts = [pts[x_order[1]],
                 pts[x_order[2]],
                 pts[x_order[3]],
                 pts[x_order[4]]]
  rank = sortperm(ordered_pts, by=pt->pt[2])
  assert
  if rank[1] == 1 || rank[1] == 4
    error("Minimum value not contained within the bracket.")
  end
  xa, fa = ordered_pts[rank[1]-1]
  xb, fb = ordered_pts[rank[1]]
  xc, fc = ordered_pts[rank[1]+1]
  return xa, xb, xc, fa, fb, fc
end

""" Check whether new_pt meets the Wolfe criteria for objective decrease
and curvature"""
function satisfies_wolfe(pt, new_pt, step_size, direction; strong=true)
  const wolfe1 = 1e-4
  const wolfe2 = 0.9
  sufficient_decrease = new_pt[2] <= pt[2] + wolfe1*step_size*direction*pt[3]
  if strong
    sufficient_curvature = abs(new_pt[3]) <= abs(wolfe2*pt[3]*direction)
  else
    sufficient_curvature = new_pt[3] >= wolfe2*pt[3]*direction
  end
  return sufficient_decrease && sufficient_curvature
end


"""Minimise a scalar function with multidimensional input"""
function minimise(f::Function,
                    x0::Vector,
                    g::Function=gradient_approximator(f;dims=length(x0));
                    method="steepest_descent",
                    max_iters=1000,
                    max_f_evals=1000,
                    x_tol=1e-8,
                    f_tol=1e-8,
                    grad_tol=1e-12,
                    constraints=[],
                    plot=false,
                    logging=false)

  tic();
  f_evals = 0; g_evals = 0; iter = 0;
  converged_dict = create_converged_dict(x_tol=x_tol, f_tol=f_tol,
      grad_tol=grad_tol)

  if plot
    if length(constraints) > 0
      x_range = constraints
    else
      x1_max = max(1, abs(x0[1])*2)
      x2_max = max(1, abs(x0[2])*2)
      x_range = [-x1_max x1_max; -x2_max x2_max]
    end
    # contour_f(x) = log(f(x))
    fig, ax1, ax2 = plot_contour(f, x_range; name=method, problem="rosenbrock")
  end

  # TODO, if approximating gradient, each g_eval == 2 * f_eval. Count this.
  pts = []
  x_log = Array(Float64,length(x0),0)
  vals_log = []
  f_evals_log = []
  x = x0
  val = f(x); f_evals += 1;
  grad = g(x); g_evals += 1
  pt = (x, val, grad)
  push!(pts, pt)

  while !converged_dict["converged"] && iter < max_iters && f_evals <= max_f_evals
    iter += 1
    if method == "steepest_descent" || iter == 1
      direction = -normalise(grad) #Steepest descent
    elseif method == "conjugate_gradients"
      # Original method
      # beta = gradient'*gradient / (gradient_prev'*gradient_prev)
      # Polak and Ribiere method
      beta = grad'*(grad-grad_prev) / (grad_prev'*grad_prev)
      direction = normalise(beta[]*direction_prev - grad)
    end

    if plot
      ax2[:plot](x[1], x[2], "o", markersize=14, markeredgewidth=1,
      markeredgecolor="w", color=(colors[iter % length(colors)+1]))
      ax2[:plot]([x[1], x[1]+.5direction[1]], [x[2], x[2]+.5direction[2]],
          "--", linewidth=1.5, color=(colors[iter % length(colors)+1]))
      if iter%100 == 0
        savefig(@sprintf "figs/%s-%s-%d.pdf" method symbol(f) iter)
      end
      sleep(.1)
    end

    if logging
      x_log = [x_log x]
      vals_log = [vals_log; val]
      f_evals_log = [f_evals_log; f_evals]
    end

    # Line search in direction with step_size α
    f_line(α) = f(x + α*direction)
    summary = line_search(f_line, 0, max_f_evals=max_f_evals-f_evals;
                          x_tol=1e-3, f_tol=1e-4, plot=false, direction=1)
    α = summary["x"]
    val = summary["min_value"]
    f_evals += summary["function_evals"]
    x = x + α*direction # get back real x
    grad = g(x); g_evals += 1
    grad_prev = copy(grad)
    direction_prev = copy(direction)
    pt = (x, val, grad)
    push!(pts, pt)
    converged_dict = convergence!(converged_dict; x_step=α, grad=grad;)
  end

  elapsed_time = toq();
  return summarise(pts, f_evals, elapsed_time; converged_dict=converged_dict,
    x_initial=x0, x_log=x_log, vals_log=vals_log, f_evals_log=f_evals_log);

end


function line_search(f::Function, x0::Number=0,
    g::Function=gradient_approximator(f);
    direction=0, x_tol=1e-8, f_tol=1e-8,
    grad_tol=1e-12, max_f_evals=100, plot=false)

  tic();

  xa, xb, xc, fa, fb, fc, pts, f_evals =  bracket(f, x0; max_evals=max_f_evals)
  bracket_f_evals = f_evals


  if plot
    fig_line, ax_line = plot_line(f,[xa, xc]; name="line")
  end

  grad = g(xb); g_evals = 1
  pt = (xb, fb, grad) # Current min point
  push!(pts, pt)

  converged_dict = create_converged_dict(x_tol=x_tol, f_tol=f_tol,
                                        grad_tol=grad_tol)

  # Search for minima
  while !converged_dict["converged"] && f_evals <= max_f_evals

    direction = grad <= 0 ? 1 : -1
    if direction > 0
      step_size = (2-ϕ)*(xc-xb)
    else
      step_size = (2-ϕ)*(xb-xa)
    end

    # Search for a good α (step size)
    while true
      if plot
        ax_line[:plot]([xa,xb,xc], [fa, fb, fc], "x", markersize=15,
         markeredgewidth=3, color=(colors[f_evals % length(colors)+1]))
        sleep(.08)
      end
      x_new = xb + step_size*direction
      f_new = f(x_new); f_evals += 1
      g_new = g(x_new); g_evals += 1
      xa, xb, xc, fa, fb, fc = rebracket(xa, xb, xc, x_new, fa, fb, fc, f_new)
      new_pt = (x_new, f_new, g_new)
      satisfies_wolfe(pt, new_pt, step_size, direction) && break
      step_size <= x_tol && break
      f_evals >= max_f_evals && break
      step_size = step_size*(2-ϕ) # Reduce step size
    end
    # println("Line search took ", f_evals,
      # " evaluations (Bracketing: ", bracket_f_evals,  ")")

    convergence!(converged_dict; x_step=step_size, grad=grad)
    grad = g(xb); g_evals += 1
    pt = (xb, fb, grad)
    push!(pts, pt)

  end
  return summarise(pts, f_evals, toq();
   converged_dict=converged_dict, x_initial=x0);
end

""" Linear conjugate gradient solver of form Ax = b"""
function conjugate_gradients(A,b=randn(size(A,1)),x=zeros(length(b));grad_tol=1e-10)
  r = b-A*x # Analytic gradient/ residual
  p = r      # Search in direction of residual (steepest descent initially)
  rs_old = r'*r
  pts = []
  pt = (x, 1, r)
  push!(pts, pt)
  for i = 1:length(b)*6
    Ap = A*p
    alpha = rs_old/(p'*Ap)
    x = x + alpha.*p
    r = r-alpha.*Ap
    rs_new = r'*r
    pt = (x,1,r)
    push!(pts, pt)
    sqrt(rs_new[]) <= grad_tol && break
    p = r+(rs_new/rs_old).*p
    rs_old = rs_new
  end
  return x
end


function mycg(A, b, x0=zeros(length(b)); grad_tol=1e-10)
  d = r = b - A*x0 # r = residual error, d = direction = -gradient (initially)
  x = copy(x0)
  for i in 1:length(b)
    Ad, rr= A*d, r'r # Precompute to save precious clock cycles

    α = rr/(d'Ad)    # α is the distance to travel along d
    x1 = x + α.*d
    r1 = r - α.*Ad
    β1 = r1'r1/(rr)  # β coefficient makes d1 conjugate (A-orthogonal) to d
    println(β1)
    β1 = r1'*(r1-r)/rr # Alternative Polak and Ribiere method
    println(β1)
    d1 = r1 + β1.*d

    r, d, x = r1, d1, x1
  end
  return x
end



# Original method
# beta = gradient'*gradient / (gradient_prev'*gradient_prev)
# Polak and Ribiere method
# beta = grad'*(grad-grad_prev) / (grad_prev'*grad_prev)

#
#   α1 =
#   d = r = b - A*x  # all fist values. d = direction, r = residual = -gradient
#   α = r'r/(d'A10*d)
#
#   for i in 1:length(b)
#     x1 = x + α.*d
#     r1 = r - α.*A*d
#     β1 = r1'r1/(r'r)
#     d1 = r1 + β1.*d
#     x, r, d = x1, r1, d1
#     println(sum(r))
#   end
#   return x
# end

# function x = mycg(A, b=randn(size(A,1)), x=zeros(length(b)), grad_tol=1e-10)
#   d = r = b - A*x  # all fist values. d = direction, r = residual = -gradient
#   a = r'r/(d'A*d)
#
#   for i in 1:length(b)
#     x1 = x + a.*d
#     r1 = r - a.*A*d
#     B1 = r1'r1/(r'r)
#     d1 = r1 + B1.*d
#     x, r, d = x1, r1, d1
#     println(sum(r))
#   end
#   return x
# end



function quadratic_conjugate_gradients(A,b=randn(size(A,1)),x0=zeros(length(b)); method="steepest_descent")
  f(x::Vector) = (1/2 * x'*A*x - b'*x)[]
  g(x::Vector) = vec(A*x - b)
  tic();
  summary = minimise(f,x0,g,method="conjugate_gradients")
  println(summary["x"])
  elapsed_time = toq();
  return summary
  # println(pts,"\n", f_evals,"\n", elapsed_time)
  # return summarise(pts, f_evals, elapsed_time)
end
