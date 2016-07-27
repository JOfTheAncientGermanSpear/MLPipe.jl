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
export plotEvalModel
export calcCorrelations


export r2score
export precisionScore
export recallScore
export f1score


export encodeCategorical


include("pipe.jl")

end
