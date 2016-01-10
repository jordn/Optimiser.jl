include("plot.jl")
include("functions.jl")
using Formatting

const ϕ = 0.5 * (1.0 + sqrt(5.0))
const global disp_progress = false
normalise(x) = x/norm(x)

function print_progress(xa, xb, xc, fa, fb, fc, evals)
  if disp_progress
    const fmt = ">{}: ({: 2.4f}, {: 2.4f}, {: 2.4f}) = ({: 2.3f}, {: 2.3f}, {: 2.3f})\n"
      printfmt(fmt, evals, xa, xb, xc, fa, fb, fc)
  end
end

"""Return a function that returns the jacobian
(2D input, 1D output only)"""
function jacobian(f::Function, δ=1e-8)
  # g(x) = [(f(x+sparsevec(Dict(i=>δ),dims)) - f(x-sparsevec(Dict(i=>δ),dims)))./(2δ) for i in 1:dims]
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

"Bracket the minimum of the function."
function bracket(f::Function, xb=0; xa=xb-(1-1/ϕ), xc=xb+1/ϕ, max_evals=10)
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
    print_progress(xa, xb, xc, fa, fb, fc, evals)

    xb, xc = xa, xb
    fb, fc = fa, fb
    push!(pts, (xb,fb,0.0))

    xa = xa - ϕ*(xc-xb) # big jump
    fa = f(xa); evals += 1;
    while fa == fb && evals <= max_evals
      print_progress(xa, xb, xc, fa, fb, fc, evals)
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
  print_progress(xa, xb, xc, fa, fb, fc, evals)
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
                     x_tolerance=0.001,
                     grad_tolerance=1e-12,
                     max_iterations=20,
                     max_f_evals=1000,
                     contraints=[],
                     plot=false)
  tic();
  f_evals = 0
  g_evals = 0
  iterations = 0
  jacobian = gradient_approximator(f,dims=length(x0))

  # PLOT
  if plot
    if length(contraints) > 0
      x_range = contraints
    else
      x1_max = max(1, abs(x0[1])*2)
      x2_max = max(1, abs(x0[2])*2)
      x_range = [-x1_max x1_max; -x2_max x2_max]
    end
    fig, ax1, ax2 = plot_contour(f, x_range; name="tabu")
  end

  # TODO, if approximating gradient, each g_eval == 2 * f_eval. Count this.
  pts = []
  x = x0
  val = f(x); f_evals += 1;

  while iterations < max_iterations && f_evals <= max_f_evals
    iterations += 1
    gradient = jacobian(x); g_evals += 1
    pt = (x, val, gradient)
    push!(pts, pt)

    if method == "steepest_descent" || iterations == 1
      direction = -normalise(gradient) #Steepest descent
    elseif method == "conjugate_gradients"
      # Original method
      # beta = gradient'*gradient / (gradient_prev'*gradient_prev)
      # Polak and Ribiere method
      beta = gradient'*(gradient-gradient_prev) / (gradient_prev'*gradient_prev)
      direction = normalise(beta[]*direction_prev - gradient)
    end

    # direction = copy(normalise(new_direction))
    # println( "direction(saagain) ", direction)

    if plot
      ax1[:plot]([x[1]], [x[2]], val, "o")
      ax2[:plot](x[1], x[2], "o--")
      ax2[:plot]([x[1], x[1]+direction[1]], [x[2], x[2]+direction[2]], "--")
      if iterations%100 == 0
        savefig(@sprintf "figs/gradient-%s-%d.png" symbol(f) iterations)
      end
      sleep(.1)
    end

    # Line search in direction with step_size α
    f_line(α) = f(x + α*direction)
    summary = line_search(f_line, 0, max_f_evals=max_f_evals-f_evals;
                          plot=false, direction=1)
    # show(summary);
    α = summary["x"]
    val = summary["min_value"]
    f_evals += summary["function_evals"]
    x = x + α*direction # get back real x
    println("step_size: ", α)
    println(x , " => ", val )
    println( "direction ", direction)
    gradient_prev = copy(gradient)
    direction_prev = copy(direction)
    println()
  end
  elapsed_time = toc();
  println(pts,"\n", f_evals,"\n", elapsed_time)
  return summarise(pts, f_evals, elapsed_time)
end


function line_search(f::Function, x0::Number=0, g::Function=gradient_approximator(f);
  direction=0, x_tolerance=0.001, grad_tolerance=1e-12, max_f_evals=20, plot=false)
  tic();
  converged(step=xc-xa) = (step <= x_tolerance
                          || abs(gradient) <= grad_tolerance)

  # TODO, save time by knowing direction, and sensible starting bracket width
  # # If we have a direction (pointing downhill) can save evaluations by initally
  # # bracketting only forwards
  # if direction > 0
  #   xa, xb, xc = x0, x0*ϕ, x0*ϕ*ϕ
  # elseif x0 != 0
  #   xa, xb, xc = x0, x0*ϕ, x0+ϕ
  # end

  # fa < fc
  xa, xb, xc, fa, fb, fc, pts, f_evals =  bracket(f, x0; max_evals=max_f_evals)

  if plot
    fig_line, ax_line = plot_line(f,[xa, xc]; name="line")
  end

  gradient = g(xb); g_evals = 1
  pt = (xb, fb, gradient) # Current min point
  push!(pts, pt)

  while f_evals <= max_f_evals && !converged()
    print_progress(xa, xb, xc, fa, fb, fc, f_evals)

    direction = gradient <= 0 ? 1 : -1 # (p_k)
    if direction > 0
      step_size = (1-(1/ϕ))*(xc-xb)
    else
      step_size = (1-(1/ϕ))*(xb-xa)
    end

    while true
      if plot
        ax_line[:plot]([xa,xb,xc], [fa, fb, fc], "o--")
        sleep(.1)
      end
      x_new = xb + step_size*direction
      f_new = f(x_new); f_evals += 1
      g_new = g(x_new); f_evals += 1
      xa, xb, xc, fa, fb, fc = rebracket(xa, xb, xc, x_new, fa, fb, fc, f_new)
      print_progress(xa, xb, xc, fa, fb, fc, f_evals)
      new_pt = (x_new, f_new, g_new)
      satisfies_wolfe(pt, new_pt, step_size, direction) && break
      converged(step_size) && break
      f_evals <= max_f_evals && break
      step_size = step_size*(1-(1/ϕ)) # Step length (α)
    end

    gradient = g(xb); g_evals += 1
    pt = (xb, fb, gradient)
    push!(pts, pt)
  end
  print_progress(xa, xb, xc, fa, fb, fc, f_evals)
  elapsed_time = toc();
  return summarise(pts, f_evals, elapsed_time);
end

# """ Linear conjugate gradient solver of form Ax = b"""
# function conjugate_gradients(A,b,x)
#   r = b-A*x
#   p = r
#   rsold = r'*r
#
#   for i = 1:length(b)
#     Ap = A*p
#     alpha = rsold/(p'*Ap)
#     x = x + alpha.*p
#     r = r-alpha.*Ap
#     rsnew = r'*r
#     if sqrt(rsnew[]) < 1e-10
#       break
#     end
#     p = r+(rsnew/rsold).*p
#     rsold = rsnew
#   end
#   return x
# end

""" Return a consistent data structure summarising the results. """
function summarise(pts, f_evals, elapsed_time=""; g_evals="")
  println("::::SUMMARY::::")
  summary = Dict{ASCIIString, Any}(
    "x" => pts[length(pts)][1],
    "min_value" => pts[length(pts)][2],
    "gradient" => pts[length(pts)][3],
    "elapsed_time" => elapsed_time,
    "function_evals" => f_evals,
    # "gradient_evals" => g_evals,
    "pts" => pts,
  )
  return summary
end
