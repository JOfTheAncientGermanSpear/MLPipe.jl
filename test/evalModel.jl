using Base.Test

include("../src/pipe.jl")
include("../src/evalModel.jl")

actual = stateCombos(:a=>[1, 2], :b=>[3, 4])
ab(a,b) = Dict(:a=>a, :b=>b)
expected = Dict[ab(1,3), ab(1, 4), ab(2, 3), ab(2, 4)]
@test actual == expected


abc(a,b,c) = Dict(:a=>a, :b=>b, :c=>c)
expected = Dict[abc(1, 3, 5), abc(1, 3, 6),
                abc(1, 4, 5), abc(1, 4, 6),
                abc(2, 3, 5), abc(2, 3, 6),
                abc(2, 4, 5), abc(2, 4, 6)]
actual = stateCombos(:a => [1, 2], :b=> [3, 4], :c=>[5, 6])
@test actual == expected
