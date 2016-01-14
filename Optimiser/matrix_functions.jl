# include("../data/data.jl")
using MAT
vars = matread("data/data.mat")

A10 = vars["A10"]
A100 = vars["A100"]
A1000 = vars["A1000"]
B10 = vars["B10"]
B100 = vars["B100"]
B1000 = vars["B1000"]
