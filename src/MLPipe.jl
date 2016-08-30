module MLPipe


export Pipeline
export pipeFit!
export pipePredict
export pipeTest


export paramState
export paramState!


export modelState
export modelState!


export calcTrainTestScores


export evalModel
export evalModelParallel
export meanTrainTest
export Combos
export ModelState
export stateCombos
export plotEvalModel
export calcCorrelations


export r2score
export precisionScore
export recallScore
export f1score


export encodeCategorical

export calcAnova
export calcCorrelations
export tukey


include("pipe.jl")
include("evalModel.jl")

end
