include("plot.jl")
include("functions.jl")

const ϕ = 0.5 * (1.0 + sqrt(5.0))
const global disp_progress = false

using Formatting

function print_progress(xa, xb, xc, fa, fb, fc, evals)
  if disp_progress
    const fmt = ">{}: ({: 2.4f}, {: 2.4f}, {: 2.4f}) = ({: 2.3f}, {: 2.3f}, {: 2.3f})\n"
      printfmt(fmt, evals, xa, xb, xc, fa, fb, fc)
  end
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
function satisfies_wolfe(pt, new_pt, step_size, direction)
  const wolfe1 = 1e-4
  const wolfe2 = 0.9
  sufficient_decrease = new_pt[2] <= pt[2] + wolfe1*step_size*direction*pt[3]
  sufficient_curvature = new_pt[3] >= wolfe2*pt[3]*direction
  return sufficient_decrease && sufficient_curvature
end

function minimise_scalar(f::Function, x0::Number, g::Function=gradient_approximator(f);
     x_tolerance=0.001, grad_tolerance=1e-12, max_evals=100)
     return minimise(f, vec([x0]), g; x_tolerance=x_tolerance,
      grad_tolerance=grad_tolerance, max_evals=max_evals)
end

function minimise(f::Function, x0::Vector, g::Function=gradient_approximator(f);
   x_tolerance=0.001, grad_tolerance=1e-12, max_evals=100, plot=false)
    tic();
    evals = 0

    println()
    # fa < fc
    xa, xb, xc, fa, fb, fc, pts, evals =  bracket(f, x0; max_evals=max_evals)
    # TODO, if approximating gradient, each g_eval == 2 * f_eval. Count this.
    gradient = g(xb); evals += 1
    pt = (xb, fb, gradient) # Current min point
    push!(pts, pt)

    converged(step=xc-xa) = (step <= x_tolerance
                            || abs(gradient) <= grad_tolerance)

    while !converged()
      print_progress(xa, xb, xc, fa, fb, fc, evals)

      direction = gradient <= 0 ? 1 : -1 # (p_k)
      if direction > 0
        step_size = (1-(1/ϕ))*(xc-xb)
      else
        step_size = (1-(1/ϕ))*(xb-xa)
      end

      while true
        x_new = xb + step_size*direction
        f_new = f(x_new); evals += 1
        g_new = g(x_new); evals += 1
        xa, xb, xc, fa, fb, fc = rebracket(xa, xb, xc, x_new, fa, fb, fc, f_new)
        print_progress(xa, xb, xc, fa, fb, fc, evals)
        new_pt = (x_new, f_new, g_new)
        satisfies_wolfe(pt, new_pt, step_size, direction) && break
        converged(step_size) && break
        step_size = step_size*(1-(1/ϕ)) # Step length (α)
      end

      if evals > max_evals
        error("Too many evaluations.")
      end
      gradient = g(xb); evals += 1
      pt = (xb, fb, gradient)
      push!(pts, pt)

    end
    print_progress(xa, xb, xc, fa, fb, fc, evals)
    elapsed_time = toq();
    return summarise(pts, evals, elapsed_time)
end

function minimise_multi(f::Function, x0, direction)
  fNew(scalar) = f(x0 + scalar*direction)
  summary = minimise_scalar(fNew, 0)
  summary["x"] -= x0
  return summary
end


""" Return a consistent data structure summarising the results. """
function summarise(pts,evals,elapsed_time)
  # println(xa, " ", xb, " ", xc, " ", fa, " ", fb, " ", fc, " ", gradient,
  # " ", evals, " ", elapsed_time)
  summary = Dict{ASCIIString, Any}(
    "x" => pts[length(pts)][1],
    "minvalue" => pts[length(pts)][2],
    "gradient" => pts[length(pts)][3],
    "elapsed_time" => elapsed_time,
    "evals" => evals,
    "pts" => pts,
  )
  return summary
end
