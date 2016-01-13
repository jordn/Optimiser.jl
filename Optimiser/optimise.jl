using Debug
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
function gradient_approximator(f::Function, δ=1e-8; dims=1)
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
function bracket(f::Function, xb=0; xa::Real=xb-(1-1/ϕ), xc::Real=xb+1/ϕ, max_evals::Int=10)
  xa, xb, xc = sort([xa, xb, xc])

  fa = f(xa)
  fc = f(xc)
  # Want fa < fc
  if fa > fc
    xa, xc = xc, xa
    fa, fc = fc, fa
  end
  # xb = xa+(ϕ-1)*(xc-xa) # Closer to a
  fb = f(xb)
  evals = 3
  pts = [(xb,fb,0.0)]
  while fb >= fa

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
      error("Too many evaluations.")
    end
  end

  if xa > xc
    xa, xc = xc, xa
    fa, fc = fc, fa
  end
  return xa, xb, xc, fa, fb, fc, pts, evals
end

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

function minimise_2d(f::Function,
                    x0::Vector,
                    g::Function=gradient_approximator(f;dims=length(x0));
                    method="steepest_descent",
                    x_tol=1e-8,
                    f_tol=1e-8,
                    grad_tol=1e-12,
                    max_iterations=100,
                    max_f_evals=1000,
                    constraints=[],
                    plot=false)
  tic();
  f_evals = 0; g_evals = 0; iterations = 0;
  converged_dict = create_converged_dict(x_tol=x_tol, f_tol=f_tol,
      grad_tol=grad_tol)

  # PLOT
  if plot
    if length(constraints) > 0
      x_range = constraints
    else
      x1_max = max(1, abs(x0[1])*2)
      x2_max = max(1, abs(x0[2])*2)
      x_range = [-x1_max x1_max; -x2_max x2_max]
    end
    fig, ax1, ax2 = plot_contour(f, x_range; name=method)
  end

  # TODO, if approximating gradient, each g_eval == 2 * f_eval. Count this.
  pts = []
  x = x0
  val = f(x); f_evals += 1;
  grad = g(x); g_evals += 1
  pt = (x, val, grad)
  push!(pts, pt)

  while !converged_dict["converged"] && iterations < max_iterations && f_evals <= max_f_evals
    iterations += 1
    if method == "steepest_descent" || iterations == 1
      direction = -normalise(grad) #Steepest descent
    elseif method == "conjugate_gradients"
      # Original method
      # beta = gradient'*gradient / (gradient_prev'*gradient_prev)
      # Polak and Ribiere method
      beta = grad'*(grad-grad_prev) / (grad_prev'*grad_prev)
      direction = normalise(beta[]*direction_prev - grad)
    end

    if plot
      ax1[:plot]([x[1]], [x[2]], val, "o")
      ax2[:plot](x[1], x[2], "o--")
      ax2[:plot]([x[1], x[1]+direction[1]], [x[2], x[2]+direction[2]], "--")
      if iterations%100 == 0
        savefig(@sprintf "figs/%s-%s-%d.pdf" method symbol(f) iterations)
      end
      sleep(.1)
    end

    # Line search in direction with step_size α
    f_line(α) = f(x + α*direction)
    summary = line_search(f_line, 0, max_f_evals=max_f_evals-f_evals;
                          plot=false, direction=1)
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

  plot_training(pts)
  elapsed_time = toq();
  return summarise(pts, f_evals, elapsed_time; converged_dict=converged_dict);
end


function line_search(
    f::Function,
    x0::Number=0,
    g::Function=gradient_approximator(f);
    direction=0,
    x_tol=1e-8,
    f_tol=1e-8,
    grad_tol=1e-12,
    max_f_evals=100,
    plot=false)

  tic();
  # TODO, save time by knowing direction, and sensible starting bracket width

  # fa < fc
  xa, xb, xc, fa, fb, fc, pts, f_evals =  bracket(f, x0; max_evals=max_f_evals)

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

    direction = grad <= 0 ? 1 : -1 # (p_k)
    if direction > 0
      step_size = (1-(1/ϕ))*(xc-xb)
    else
      step_size = (1-(1/ϕ))*(xb-xa)
    end

    # Search for optimal α (step size)
    while true
      if plot
        ax_line[:plot]([xa,xb,xc], [fa, fb, fc], "o--")
        sleep(.1)
      end
      x_new = xb + step_size*direction
      f_new = f(x_new); f_evals += 1
      g_new = g(x_new); f_evals += 1
      xa, xb, xc, fa, fb, fc = rebracket(xa, xb, xc, x_new, fa, fb, fc, f_new)
      new_pt = (x_new, f_new, g_new)
      satisfies_wolfe(pt, new_pt, step_size, direction) && break
      f_evals <= max_f_evals && break
      step_size = step_size*(1-(1/ϕ)) # Step length (α)
    end
    convergence!(converged_dict; x_step=step_size, grad=grad)
    grad = g(xb); g_evals += 1
    pt = (xb, fb, grad)
    push!(pts, pt)

  end
  elapsed_time = toq();
  return summarise(pts, f_evals, elapsed_time;
   converged_dict=converged_dict);
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


@debug function mycg(A, b, x0=zeros(length(b)); grad_tol=1e-10)
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
  summary = minimise_2d(f,x0,g,method="conjugate_gradients")
  println(summary["x"])
  elapsed_time = toq();
  return summary
  # println(pts,"\n", f_evals,"\n", elapsed_time)
  # return summarise(pts, f_evals, elapsed_time)
end
