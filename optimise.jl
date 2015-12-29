using Formatting
using PyPlot
using Optim

const ϕ = 0.5 * (1.0 + sqrt(5.0))
const global disp_progress = true
f(x) = x.^4 .* cos(1./x) + 2x.^4
g(x) = 4x.^3 .* cos(1./x) + x.^2 .* sin(1./x) + 8x.^3
f(x) = x.^2-10x
g(x) = 2x-10
x = collect(-0.1:0.000001881:0.1)
plot(x,f(x))

# plot(-6:0.1:6, f(-6:0.1:6))


function print_progress(xa, xb, xc, fa, fb, fc, evals)
  if disp_progress
    const fmt = "{}: ({: 2.2f}, {: 2.2f}, {: 2.2f}) = ({: 2.3f}, {: 2.3f}, {: 2.3f})\n"
    # if xa > xc
      # printfmt(fmt, evals, xc, xb, xa, fc, fb, fa)
      # @printf "%d: (% 2.2f, % 2.2f, % 2.2f) = (% 2.3f, % 2.3f, % 2.3f)\n" evals xc xb xa fc fb fa
    # else
      # @printf "%d: (% 2.2f, % 2.2f, % 2.2f) = (% 2.3f, % 2.3f, % 2.3f)\n" evals xa xb xc fa fb fc
      printfmt(fmt, evals, xa, xb, xc, fa, fb, fc)
    # end
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

  while fb >= fa
    print_progress(xa, xb, xc, fa, fb, fc, evals)

    xb, xc = xa, xb
    fb, fc = fa, fb

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
  println("BRACKETED")
  return xa, xb, xc, fa, fb, fc, evals
end

# xa, xb, xc, fa, fb, fc, evals =  bracket(f)
# xa, xb, xc, fa, fb, fc, evals =  bracket(f,1,-20,21)
# xa, xb, xc, fa, fb, fc, evals =  bracket(f,1,6,111)

function rebound(xa, xb, xc, x_new, fa, fb, fc, f_new)
  pts = [(xa, fa), (xb, fb), (xc, fc), (x_new, f_new)]
  x_order = sortperm(pts, by=pt->pt[1])
  ordered_pts = [pts[x_order[1]], pts[x_order[2]], pts[x_order[3]], pts[x_order[4]]]
  rank = sortperm(ordered_pts, by=pt->pt[2])
  if rank[1] == 1 || rank[1] == 4
    error("Fucked up the bracketing")
  end
  xa, fa = ordered_pts[rank[1]-1]
  xb, fb = ordered_pts[rank[1]]
  xc, fc = ordered_pts[rank[1]+1]
  return xa, xb, xc, fa, fb, fc
end


function minimise(f::Function, x0, g=None; x_tolerance=0.01, max_evals=30)
    tic();
    const ϵ = 1e-10
    const fmt = "{}: ({: 2.2f}, {: 2.2f}, {: 2.2f}) = ({: 2.3f}, {: 2.3f}, {: 2.3f})\n"
    evals = 0

    # fa < fc
    xa, xb, xc, fa, fb, fc, evals =  bracket(f, x0; max_evals=max_evals-evals)
    gradient = g(xb); evals += 1
    while xc-xa > x_tolerance && abs(gradient) >= ϵ
      println();
      f_min = fb
      print_progress(xa, xb, xc, fa, fb, fc, evals)
      direction = gradient <= 0 ? 1 : -1 # (p_k)
      if direction > 0
        step_size = (1-(1/ϕ))*(xc-xb)
      else
        step_size = (1-(1/ϕ))*(xb-xa)
      end
      x_new = xb + step_size*direction
      f_new = f(x_new); evals += 1
      g_new = g(x_new); evals += 1
      @printf "New pt: %0.3f => %0.3f   /%0.3f/\n" x_new f_new g_new

      xa, xb, xc, fa, fb, fc = rebound(xa, xb, xc, x_new, fa, fb, fc, f_new)

      function satisfies_wolfe()
        const wolfe_const1 = 1e-4
        const wolfe_const2 = 0.9
        sufficient_decrease = f_new <= f_min + wolfe_const1 * step_size* gradient * direction
        sufficient_curvature = g_new >= wolfe_const2 * gradient * direction
        return sufficient_decrease && sufficient_curvature
      end

      # printfmt(fmt, evals, xa, xb, xc, fa, fb, fc)
      print_progress(xa, xb, xc, fa, fb, fc, evals)
      while !satisfies_wolfe() && step_size > x_tolerance && abs(g_new) >= ϵ
        step_size = step_size*(1-(1/ϕ)) # Step length (α)
        x_new = xb + step_size*direction
        f_new = f(x_new); evals += 1
        g_new = g(x_new); evals += 1
        @printf "New pt: %0.3f => %0.3f   /%0.3f/\n" x_new f_new g_new
        xa, xb, xc, fa, fb, fc = rebound(xa, xb, xc, x_new, fa, fb, fc, f_new)

        print_progress(xa, xb, xc, fa, fb, fc, evals)
      end
      println("leaving wolfe loop")

      if evals > max_evals
        error("Too many evaluations.")
      end
      f_min = fb
      gradient = g(xb); evals += 1

    end
    print_progress(xa, xb, xc, fa, fb, fc, evals)
    elapsed_time = toc();

    return summarise(xa, xb, xc, fa, fb, fc, evals, elapsed_time)
end


""" Return a consistent data structure summarising the results. """
function summarise(xa,xb,xc,fa,fb,fc,evals,elapsed_time)
  return xb
end


# tic()
# res = optimize(f, -4.0,45)
# println(res)
# toc()


⭐ = minimise(f, 20, g)
println(⭐)
