include("../data/data.jl")
srand(567)


bA2 = randn(2)
A2 = A10[1:2,1:2]
fA2(x::Vector) = (1/2 * x'*A2*x - bA2'*x)[]

bA10 = randn(10)
fA10(x::Vector) = (1/2 * x'*A10*x - bA10'*x)[]

bB10 = randn(10)
fB10(x::Vector) = (1/2 * x'*B10*x - bB10'*x)[]

bA100 = randn(100)
fA100(x::Vector) = (1/2 * x'*A100*x - bA100'*x)[]

bB100 = randn(100)
fB100(x::Vector) = (1/2 * x'*B100*x - bB100'*x)[]
