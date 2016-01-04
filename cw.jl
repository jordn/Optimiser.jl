using Optim
using Gadfly

include("Optimiser/optimise.jl")

f(x) = x.^4 .* cos(1./x) + 2x.^4
g(x) = 4x.^3 .* cos(1./x) + x.^2 .* sin(1./x) + 8x.^3
# f(x) = x.^2-10x
# g(x) = 2x-10
x = collect(-0.1:0.000001881:0.1)
# plot(x,f(x))
# plot(-6:0.1:6, f(-6:0.1:6))
# xa, xb, xc, fa, fb, fc, evals =  bracket(f)
# xa, xb, xc, fa, fb, fc, evals =  bracket(f,1,-20,21)
# xa, xb, xc, fa, fb, fc, evals =  bracket(f,1,6,111)

# tic()
# res = optimize(f, -4.0,45)
# toc()
# println(res)
# @time res = optimize(f, -4.0,45)

@time summary = minimise(f, 20, g)
pts = summary["pts"]
array = zeros(length(pts), 3)
for i = 1:length(pts)
  array[i, 1] = pts[i][1]
  array[i, 2] = pts[i][2]
  array[i, 3] = pts[i][3]
end

# p = plot(x=randn(2000), Geom.histogram(bincount=100))
# draw(PDF("equation-graph.pdf", 16cm, 12cm), p)

rosenbrock(x,y) = (1-x)^2+100(y-x^2)^2

summary = minimise_multi(rosenbrock , [2, 2], [0, 1])
pts = summary["pts"]
array = zeros(length(pts), 3)
for i = 1:length(pts)
  array[i, 1] = pts[i][1]
  array[i, 2] = pts[i][2]
  array[i, 3] = pts[i][3]
end
