# using Debug
include("plot.jl")
include("functions.jl")
include("matrix_functions.jl")
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
    # @printf "(%0.2f, %0.2f, %0.2f) = %0.2f, %0.2f, %0.2f\n" xa xb xc fa fb fc
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

  pts = []
  x = copy(x0)
  val = f(x); f_evals += 1;
  grad = g(x); g_evals += 1
  pts = [pts; (x, val, grad)]
  x_log = Array(Float64,length(x0),0)
  vals_log = []
  grad_log = Array(Float64,length(grad),0)
  f_evals_log = []

  direction = -grad

  while !converged_dict["converged"] && iter < max_iters && f_evals <= max_f_evals
    iter += 1

    if plot
      ax2[:plot](x[1], x[2], "o", markersize=14, markeredgewidth=1,
      markeredgecolor="w", color=(colors[iter % length(colors)+1]))
      arrow = 0.5 * normalise(direction)
      ax2[:plot]([x[1], x[1]+arrow[1]], [x[2], x[2]+arrow[2]],
          "--", linewidth=1.5, color=(colors[iter % length(colors)+1]))
      if iter < 10 || iter%20 == 0
        savefig(@sprintf "figs/m%s-%s-%04d.pdf" method symbol(f) iter)
      end
      sleep(.1)
    end

    if logging
      x_log = [x_log x]
      vals_log = [vals_log; val]
      grad_log = [grad_log grad]
      f_evals_log = [f_evals_log; f_evals]
    end

    grad = g(x); g_evals += 1
    direction = -grad

    # Line search in direction d with step_size α
    f_line(α) = f(x + α*direction)
    line_summary = line_search(f_line, 0, max_f_evals=max_f_evals-f_evals;
                          x_tol=1e-6, f_tol=1e-4, plot=false, direction=1)
    println(line_summary)
    α = line_summary["x"]
    val = line_summary["min_value"]
    f_evals += line_summary["function_evals"]

    x = x + α.*direction        # get back real x
    #
    # grad1 = g(x1); g_evals += 1
    #
    # if method == "conjugate_gradients"
    #   r1 = -grad1
    #   β1 = r1'r1/(r'r)
    #   d1 = r1 + β1.*direction
    # elseif method == "steepest_descent"
    #   d1 = -grad1
    # end
    # direction = d1
    # grad, x = grad1, x1
    push!(pts, (x, val, grad))
    converged_dict = convergence!(converged_dict; x_step=α, grad=grad;)
  end

  elapsed_time = toq();
  return summarise(pts, f_evals, elapsed_time; converged_dict=converged_dict,
    x_initial=x0, x_log=x_log, vals_log=vals_log, f_evals_log=f_evals_log,
    method=method, grad_log=grad_log);
end

""" Perform a one dimensional line search by golden section """
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
      println(x_new)
    end
    println("Line search took ", f_evals,
      " evaluations (Bracketing: ", bracket_f_evals,  ")")

    convergence!(converged_dict; x_step=step_size, grad=grad)
    grad = g(xb); g_evals += 1
    pt = (xb, fb, grad)
    push!(pts, pt)

  end
  if f_evals < 10 || f_evals %20 == 0
    savefig(@sprintf "figs/line%s-%04d.pdf" symbol(f) f_evals)
  end
  return summarise(pts, f_evals, toq(); method=:line_search,
   converged_dict=converged_dict, x_initial=x0);
end


""" Conjugate gradient solver of quadratic forms"""
function conjgrad(A, b, x0=zeros(length(b));
   max_f_evals=1000, grad_tol=1e-10, method=:polak)

  tic(); pts = []; f_evals=0;
  f(x) = (1/2*x'A*x - b'x)[] # Treat as scalar rather than 1x1 matrix

  if method == :polak
    β(r, r1) = r1'*(r1-r)/r'r # Polak and Ribiere method, often performs better
  else
    β(r, r1) = r1'r1/(r'r)    # The 'classic'
  end

  x = copy(x0)
  d = r = b - A*x # r = residual error, d = direction = -gradient (initially)

  while abs(norm(d)) > grad_tol || f_evals <= 40
    pts = [pts; (x, f(x), -d)];
    f_evals += 1
    Ad = A*d        # Precompute to save precious clock cycles

    α = r'r/(d'Ad)  # α is the "exact" distance to travel along d
    x1 = x + α.*d
    r1 = r - α.*Ad
    β1 = β(r,r1)    # β coefficient makes d1 conjugate (A-orthogonal) to d
    d1 = r1 + β1.*d

    r, d, x = r1, d1, x1
  end

  return summarise(pts, f_evals, toq(), method=method, x_initial=x0)
end


""" Steepest descent solver of quadratic forms"""
function steepdesc(A, b, x0=zeros(length(b));
   max_f_evals=1000, grad_tol=1e-10)

  converged_dict = create_converged_dict(grad_tol=grad_tol)
  tic(); pts = []; f_evals=0;
  f(x) = (1/2*x'A*x - b'x)[] # Treat as scalar rather than 1x1 matrix
  x = copy(x0)

  while true
    r = b - A*x      # r = residual error = -gradient
    α = r'r/(r'A*r)  # α is distance s.t. gradient at next x is orthogonal
    x = x + α.*r

    pts = [pts; (x, f(x), -r)];
    f_evals += 1
    convergence!(converged_dict;  grad=norm(r))
    converged_dict["converged"] && break
    f_evals > max_f_evals && break
  end

  return summarise(pts, f_evals, toq(), converged_dict=converged_dict,
  x_initial=x0)
end



""" Steepest descent solver of quadratic forms"""
function goldensection(A, b, x0=zeros(length(b)))
  f(x::Vector) = (1/2 * x'*A*x - b'*x)[]
  g(x::Vector) = vec(A*x - b)
  tic();

  summary = minimise(f, x0, g, method="steepest_descent", max_f_evals=5000)
  println(summary["x"])
  elapsed_time = toq();
  return summary

end
