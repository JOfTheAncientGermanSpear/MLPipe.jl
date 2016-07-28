using DataFrames

include("anova.jl")

harmonicmean{T <: Real}(ns::AbstractVector{T}) = length(ns)/sum(1./ns)

#http://web.mst.edu/~psyworld/anovaexample.htm
#http://web.mst.edu/~psyworld/tukeysexample.htm#1
function tukey{T}(labels::AbstractVector{T},
               df::Int64,
               sumSqrs::Float64,
               means::AbstractVector{Float64},
               ns::AbstractVector{Int64})
  ms_within::Float64 = sumSqrs / df
  n::Float64 = harmonicmean(ns)
  mean_error::Float64 = sqrt(ms_within/n)

  num_mns::Int64 = length(means)
  qs = Float64[]
  ps = Float64[]
  lefts = T[]
  rights = T[]
  for ix in 1:(num_mns - 1)
    for ix2 in (ix+1):num_mns
      ((right, right_mean), (left, left_mean)) = sort(
        [(labels[i], means[i]) for i in [ix, ix2]], by=im -> im[2])

      q::Float64 = (left_mean - right_mean)/mean_error
      qs = [qs; q]
      ps = [ps; ccdf(TDist(df), q)]

      lefts = [lefts; left]
      rights = [rights; right]
    end
  end

  DataFrame(left=lefts, right=rights, q=qs, pval=ps)
end


function tukey(ai::AnovaInfo)
  labels::AbstractVector = ai.groups_info[:label]
  ns::AbstractVector{Int64} = ai.groups_info[:n]
  means::AbstractVector{Float64} = ai.groups_info[:mean]

  wg::AbstractVector{Bool} = ai.results_info[:source] .== "Within Group"
  df::Int64 = ai.results_info[wg, :df][1]
  sumSqrs::Float64 = ai.results_info[wg, :sum_squares][1]

  tukey(labels, df, sumSqrs, means, ns)
end
