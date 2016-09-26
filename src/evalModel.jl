using MLBase

function trainTestPreds(pipe::Pipeline, cvg::CrossValGenerator)
  num_iterations = length(cvg)
  num_samples = length(pipe.truths)

  preds = zeros(Float64, num_samples)
  test_counts = zeros(Int64, num_samples)

  train_scores = zeros(Float64, num_iterations)
  fit_call = 0

  function fit(ixs::IXs)
    fit_call += 1
    pipeFit!(pipe, ixs)
    train_scores[fit_call] = pipeTest(pipe, ixs)
  end

  function test(_, ixs::IXs)
    test_counts[ixs] += 1
    preds[ixs] = (pipePredict(pipe, ixs) + (test_counts[ixs] - 1) .* preds[ixs])./test_counts[ixs]
    pipeTest(pipe, ixs)
  end

  test_scores = cross_validate(fit, test, num_samples, cvg)
  train_scores, test_scores, preds
end

typealias EvalInput{T <: AbstractVector} Pair{Symbol, T}
typealias Combos Vector{ModelState}
function stateCombos(ei...)

  #enumerate hack to keep ordering
  need_splits::Vector{Int64} = @>> enumerate(ei) begin
    filter( ix_fv::Tuple{Int64, EvalInput} -> length(ix_fv[2][2]) > 1)
    map( ix_fv -> ix_fv[1])
  end

  if length(need_splits) == 0
    ms::ModelState = ModelState(k => v[1] for (k, v) in ei)
    Combos([ms])
  else
    ret = Combos()

    ix::Int64 = need_splits[1]
    f::Symbol, vs::AbstractVector = ei[ix]
    for v in vs
      remaining::Vector{EvalInput} = begin
        r = EvalInput[l for l in identity(ei)]
        r[ix] = f => [v]
        r
      end

      ret =[ret; stateCombos(remaining...)]
    end

    ret
  end
end


meanTrainTest{T <: AbstractVector}(train::T, test::T) = mean(train), mean(test)


doNothing(train_scores::Vector, test_scores::Vector, preds::Vector, combo_ix::Int64) = ()


function evalModel(pipe::Pipeline, cvg::CrossValGenerator,
    on_combo_complete::Function=doNothing,
    states...)
  state_combos::Combos = stateCombos(states...)
  evalModel(pipe, cvg, state_combos, on_combo_complete=on_combo_complete)
end


function evalModel(pipe::Pipeline, cvg::CrossValGenerator,
    state_combos::Combos;
    on_combo_complete::Function=doNothing)

  scores::Vector{Tuple{Float64, Float64}} = map(state_combos |> enumerate) do comboix_combo
    combo_ix::Int64, combo::ModelState = comboix_combo
    modelState!(pipe, combo)
    train_scores, test_scores, preds = trainTestPreds(pipe, cvg)

    on_combo_complete(train_scores, test_scores, preds, combo_ix)

    meanTrainTest(train_scores, test_scores)
  end

  train_scores = Float64[t[1] for t in scores]
  test_scores = Float64[t[2] for t in scores]

  train_scores, test_scores, state_combos
end


stringifyLabels(labels::Vector{ModelState}) = map(labels) do m::ModelState
  @> map(p::ParamState -> "$(p[1]): $(p[2])", m) join("; ")
end


function scoresLayer(scores, clr, x; include_smooth=true)
  color = "colorant\"$clr\"" |> parse |> eval
  geoms = include_smooth ? (Geom.point, Geom.smooth) : (Geom.point,)
  layer(y=scores, x=x, Theme(default_color=color), geoms...)
end


function plotPreds(truths, preds::Vector, subjects)
  perm_ixs = sortperm(truths)

  plot(scoresLayer(truths[perm_ixs], "deepskyblue", subjects[perm_ixs]),
   scoresLayer(preds[perm_ixs], "green", subjects[perm_ixs], include_smooth=false),
   Guide.xlabel("Subject"),
   Guide.ylabel("Score"))
end

function plotPreds(truths, model_eval, subjects)
  best_score_ix = model_eval[2] |> sortperm |> last
  preds = model_eval[4][:, best_score_ix]
  plotPreds(truths, preds, subjects)
end


function plotEvalModel(train_scores, test_scores, labels)
  plot(scoresLayer(train_scores, "deepskyblue", labels),
       scoresLayer(test_scores, "green", labels),
       Guide.xlabel("Model State"),
       Guide.ylabel("Score"))
end

typealias Scores Vector{Float64}
function plotEvalModel{T}(modelEval::Tuple{Scores, Scores, Vector{T}})
  mkLabel(p::ParamState) = "$(p[1]): $(p[2])"
  mkLabel(m::ModelState) = join(map(mkLabel, m), "; ")
  labels = map(mkLabel, modelEval[3])
  plotEvalModel(modelEval[1], modelEval[2], labels)
end
