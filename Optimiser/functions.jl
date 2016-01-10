multimin(x::Number) = x.^4 .* cos(1 ./ x) + 2 .*x.^4
multimin_grad(x) = 4x.^3 .* cos(1./x) + x.^2 .* sin(1./x) + 8x.^3

# Takes in two 1D arrays and creates a 2D grid_size
rosenbrock(x::Vector) = (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
logrosenbrock(x::Vector) = log(rosenbrock(x))

# Takes a matrix where each column is an input, returns a vector
# rosenbrock{T<:Number}(X::Array{T,2}) = vec(rosenbrock(X[1,:], X[2,:]))
# Takes a vector, returns a number

camel(x::Vector) = (4 - 2.1 *x[1]^2 + (1/3)*x[1]^4)*x[1]^2 + x[1]*x[2] + (4 * x[2]^2 - 4)*x[2]^2
logcamel(x::Vector) = log(camel(x)+1.05)
camel{T<:Number}(X::Array{T,2}) = [camel(X[1,i], X[2,i]) for i in 1:size(X,2)]
