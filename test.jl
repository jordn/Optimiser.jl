using Base.Test
include("Optimiser/optimise.jl")


f(x) = x^2 # minimum = 0
g(x) = 2x
xa, xb, xc, fa, fb, fc, pts, evals = bracket(f, 10)
@test xa < 0
@test 0 < xc

g_approx = gradient_approximator(f)
@test 3.9 < g_approx(2) < 4.1

# f(X) = (10 - X[1, 1])^2 + (0 - X[1, 2])^2 + (0 - X[2, 1])^2 + (5 - X[2, 2])^2
# f([1 2; 3 1])
#
# function g!(X, S)
#     S[1, 1] = -20 + 2 * X[1, 1]
#     S[1, 2] = 2 * X[1, 2]
#     S[2, 1] = 2 * X[2, 1]
#     S[2, 2] = -10 + 2 * X[2, 2]
#     return
# end
#
# A = [0 0; 0 0]
# g!([1 2;3 1], A)
# A
