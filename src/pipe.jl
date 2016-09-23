using DataFrames
using Gadfly
using Lazy
using MLBase

include("./anova/anova.jl")
include("./anova/posthoc.jl")

typealias Functions Vector{Function}

type Pipeline
  fits::Functions
  predicts::Functions
  score_fn::Function
  truths::AbstractVector
  param_state::Function
  param_state!::Function

  Pipeline(fits::Functions, predicts::Functions,
                    score_fn::Function, truths::AbstractVector,
                    param_state::Function, param_state!::Function) = new(
      fits, predicts, score_fn, truths, param_state, param_state!)
end


typealias ParamState{T <: Any} Pair{Symbol, T}
typealias ModelState Dict{Symbol, Any}
paramState!(pipe::Pipeline, p::ParamState) = pipe.param_state!(p)
modelState!(pipe::Pipeline, m::ModelState) = for p in m
  paramState!(pipe, p)
end

paramState(pipe::Pipeline, s::Symbol) = pipe.param_state(s)
modelState(pipe::Pipeline, ps::Vector{Symbol}) = ParamState[
                                                  p => paramState(pipe, p)
                                                  for p in ps]


function Pipeline{T <: Any}(fits::Functions, predicts::Functions,
                  score_fn::Function, truths::AbstractVector,
                  model_state::Dict{Symbol, T})

  paramState!(p::ParamState) = model_state[p[1]] = p[2]

  paramState(s::Symbol) = model_state[s]

  Pipeline(fits, predicts, score_fn, truths, paramState, paramState!)
end


typealias IXs AbstractVector{Int64}

_runFns(p::Pipeline, f::Symbol, ixs::IXs, stop_fn::Int64) = foldl(
  ixs, getfield(p, f)[1:stop_fn]) do prev_out, fn::Function
    fn(prev_out)
end

_runFns(p::Pipeline, f::Symbol, ixs::IXs) = _runFns(
  p, f, ixs, length(getfield(p, f)))


function _runFns(p::Pipeline, f::Symbol, x_ixs::IXs, y_ixs::IXs, stop_fn::Int64)
  fs = getfield(p, f)
  first_out = fs[1](x_ixs, y_ixs=y_ixs)
  foldl( first_out, fs[2:stop_fn]) do prev_out, fn::Function
    fn(prev_out)
  end
end

_runFns(p::Pipeline, f::Symbol, x_ixs::IXs, y_ixs::IXs) = _runFns(
  p, f, x_ixs, y_ixs, p.(f) |> length)


pipeFit!(pipe::Pipeline, ixs::IXs) = _runFns(pipe, :fits, ixs)
pipeFit!(pipe::Pipeline, ixs::IXs, stop_fn::Int64) = _runFns(
  pipe, :fits, ixs, stop_fn)

pipeFit!(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs) = _runFns(
  pipe, :fits, x_ixs, y_ixs)
pipeFit!(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs, stop_fn::Int64) = _runFns(
  pipe, :fits, x_ixs, y_ixs, stop_fn)



pipePredict(pipe::Pipeline, ixs::IXs) = _runFns(pipe, :predicts, ixs)
pipePredict(pipe, ixs::IXs, stop_fn::Int64) = _runFns(
  pipe, :predicts, ixs, stop_fn)

pipePredict(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs) = _runFns(
  pipe, :predicts, x_ixs, y_ixs)
pipePredict(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs, stop_fn::Int64) = _runFns(
  pipe, :predicts, x_ixs, y_ixs, stop_fn)


function pipeTest(pipe::Pipeline, ixs::IXs)
  truths = pipe.truths[ixs]
  preds = pipePredict(pipe, ixs)
  pipe.score_fn(truths, preds)
end

function pipeTest(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs)
  truths = pipe.truths[x_ixs]
  preds = pipePredict(pipe, x_ixs, y_ixs)
  pipe.score_fn(truths, preds)
end


sqDist(x, y) = norm( (y - x).^2, 1)

function r2score{T <: Real}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})

  dist_from_pred::Float64 = sqDist(y_true, y_pred)
  dist_from_mean::Float64 = sqDist(y_true, mean(y_true))

  1 - dist_from_pred/dist_from_mean
end


precisionScore(t_pos::Int64, f_pos::Int64) = t_pos/(t_pos + f_pos)

function precisionScore(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  true_pos = sum(y_pred & y_true)
  false_pos = sum(y_pred & !y_true)

  precisionScore(true_pos, false_pos)
end

recallScore(t_pos::Int64, f_neg::Int64) = t_pos/(t_pos + f_neg)

function recallScore(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  true_pos = sum(y_pred & y_true)
  false_neg = sum(!y_pred & y_true)

  recallScore(true_pos, false_neg)
end


function MLBase.f1score(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  true_pos = sum(y_true & y_pred)
  if true_pos == 0
    return 0
  end

  precision = precisionScore(y_true, y_pred)
  recall = recallScore(y_true, y_pred)

  2 * precision * recall / (precision + recall)
end


function MLBase.f1score{T}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})
  num_samples = length(y_true)
  labels = unique(y_true)

  reduce(0., labels) do acc::Float64, l::T
    truths = y_true .== l
    preds = y_pred .== l
    score = f1score(truths, preds)

    weight = sum(truths)/num_samples
    acc + score * weight
  end
end


function accuracy{T <: Real}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})
  @assert length(y_true) == length(y_pred)
  sum(y_true .== y_pred)/length(y_true)
end


isCategorical{T <: AbstractString}(arr::AbstractArray{T}) = true
isCategorical(arr) = false

encodeCategorical(arr) = labelencode(labelmap(arr), arr)


function calcAnova(dataf::DataFrame,
  predictors::Vector{Symbol},
  prediction::Symbol)

  preds = dataf[prediction]
  reduce(Dict{Symbol, AnovaInfo}(), predictors) do ret, p
    genDataGroup(label) = DataGroup(dataf[preds .== label, p], label)

    datagroups = map(genDataGroup, preds |> unique)
    ret[p] = calcanova(datagroups...)
    ret
  end
end

function calcCorrelations(dataf::DataFrame, predictors::Vector{Symbol},
  prediction::Symbol)

  cors = map(p -> cor(dataf[p], dataf[prediction]), predictors)
  ret = DataFrame(predictor = predictors, cor = cors[:])
  sort(ret, cols=[:cor])
end
