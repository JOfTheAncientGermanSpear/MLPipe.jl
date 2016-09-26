@everywhere using MLBase
@everywhere using Lazy
@everywhere using PyCall
@everywhere using RDatasets

@everywhere using MLPipe

#julia -p 3
#julia> include("parallelExample") #sometimes need to run twice
#julia> model_eval = runExample()
#julia> plotEvalModel(model_eval)


@everywhere LinearSVC = begin
  @pyimport sklearn.svm as svm
  svm.LinearSVC
end


@everywhere iris = dataset("datasets", "iris");
@everywhere iris[:SpeciesInt] = encodeCategorical(iris[:Species])

@everywhere predictor_cols = names(iris)[1:4]

#make shared array to facilitate parallel processing

X_data, y_data, X_validation, y_validation = begin
  ##make shared array so that accessable from multiple processes
  y = convert(SharedArray{Int64}, Vector(iris[:SpeciesInt]))
  X = convert(SharedArray{Float64}, Matrix(iris[predictor_cols]))

  num_subjects = size(iris, 1)
  validation_ixs = 1:5:num_subjects
  data_ixs = setdiff(1:num_subjects, validation_ixs)

  X[data_ixs, :], y[data_ixs], X[validation_ixs, :], y[validation_ixs]
end


@everywhere function pipelineGen(X, y)
  #dictionary to hold model state during processing
  #updated over range of Cs by evalModel, will be clear shortly
  model_state = Dict(:svc_C => 1.)

  println("a new pipeline is generated for each combo")

  svc = LinearSVC()

  #fit functions
  fits = begin
    getXy(ixs) = X[ixs, :], y[ixs]

    function svcFit!(Xy)
      svc[:C] = model_state[:svc_C]
      svc[:fit](Xy[1], Xy[2])
    end

    [getXy, svcFit!]
  end

  #predict functions
  predicts = begin
    getX(ixs) = X[ixs, :]

    svcPredict(X) = svc[:predict](X)

    [getX, svcPredict]
  end

  truths = y |> Vector{Int64}

  ###Create Pipeline
  Pipeline(fits, predicts, f1score, truths, model_state)
end

cvgGen(y) = StratifiedRandomSub(y, round(Int64, length(y)*.8), 10)


function evalModelParallel(X, y, pipe_factory::Function, cvg_factory::Function,
    state_combos::Combos)

  scores = pmap(state_combos) do c
    trains, tests, model = evalModel(pipe_factory(X, y), cvg_factory(y), [c])
    trains[1], tests[1], model[1]
  end

  train_scores = Float64[s[1] for s in scores]
  test_scores = Float64[s[2] for s in scores]
  combos = ModelState[s[3] for s in scores]

  train_scores, test_scores, combos
end


runExample() = evalModelParallel(X_data, y_data,
    pipelineGen, cvgGen,
    stateCombos(:svc_C => [1e-3, 5e-3, 1e-2, .1, 1, 10, 50, 1e2, 5e2]))
