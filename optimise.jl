using Debug
using Formatting
# using PyPlot
import DataStructures: OrderedDict


const ϕ = 0.5 * (1.0 + sqrt(5.0))
const global disp_progress = true
# const ϕ = 1 + 0.5 * (1.0 - sqrt(5.0))
f(x) = x.^2-10x
g(x) = 2x-10
# plot(-6:0.1:6, f(-6:0.1:6))


function print_progress(xa, xb, xc, fa, fb, fc, evals)
  if disp_progress
    const fmt = "{}: ({: 2.2f}, {: 2.2f}, {: 2.2f}) = ({: 2.3f}, {: 2.3f}, {: 2.3f})\n"
    if xa > xc
      printfmt(fmt, evals, xc, xb, xa, fc, fb, fa)
      # @printf "%d: (% 2.2f, % 2.2f, % 2.2f) = (% 2.3f, % 2.3f, % 2.3f)\n" evals xc xb xa fc fb fa
    else
      # @printf "%d: (% 2.2f, % 2.2f, % 2.2f) = (% 2.3f, % 2.3f, % 2.3f)\n" evals xa xb xc fa fb fc
      printfmt(fmt, evals, xa, xb, xc, fa, fb, fc)
    end
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

  return xa, xb, xc, fa, fb, fc, evals
end

# xa, xb, xc, fa, fb, fc, evals =  bracket(f)
# xa, xb, xc, fa, fb, fc, evals =  bracket(f,1,-20,21)
# xa, xb, xc, fa, fb, fc, evals =  bracket(f,1,6,111)

""" Return a consistent data structure summarising the results. """
function summarise(xa,xb,xc,fa,fb,fc,evals,elapsed_time)
  return xb
end

function minimize(f::Function, x0, g=None; x_tolerance=0.01, max_evals=30)
    tic();
    const ϵ = 1e-20
    evals = 0
    xa, xb, xc, fa, fb, fc, evals =  bracket(f, x0; max_evals=max_evals-evals)
    gradient = g(xb); evals += 1
    while xc-xa > x_tolerance && abs(gradient) >= ϵ
      print_progress(xa, xb, xc, fa, fb, fc, evals)
      α = 0.5*(xc-xa)
      x_new = xb - α*gradient
      f_new = f(x_new); evals += 1
      @printf "(% 2.2f) => (% 2.2f) = % 2.3f \n" gradient x_new f_new

      if f_new < fb
        if x_new > xb
          xa, xb = xb, x_new
          fa, fb = fb, f_new
        else
          xb, xc = x_new, xb
          fb, fc = f_new, fb
        end
      else
        if x_new > xb
          xc, fc = x_new, f_new
        else
          xa, fa = x_new, f_new
        end
      end
      if evals > max_evals
        error("Too many evaluations.")
      end
      gradient = g(xb); evals += 1
    end
    print_progress(xa, xb, xc, fa, fb, fc, evals)
    elapsed_time = toc();

    return summarise(xa, xb, xc, fa, fb, fc, evals, elapsed_time)
end

⭐ = minimize(f, 500, g)
println(⭐)
