using Base
using DataFrames
using Distributions
using HypothesisTests
using Lazy

immutable DataGroup
  data::Vector
  sample_size::Int64
  sample_mean::Float64
  sample_std::Float64
  sample_stde::Float64
  weighted_mean::Float64
  sum_squares::Float64
  label::Union{Symbol, AbstractString}
end

function DataGroup(data::AbstractVector, label::Union{AbstractString, Symbol})
  sample_size::Int64 = length(data)
  sample_mean::Float64 = mean(data)
  sample_std::Float64 = std(data)
  sample_stde::Float64 = sample_std/sqrt(sample_size)
  weighted_mean::Float64 = sample_size * sample_mean
  sum_squares::Float64 = sample_std^2 * (sample_size - 1)

  DataGroup(data, sample_size, sample_mean, sample_std, sample_stde, weighted_mean, sum_squares, label)
end


pluckGen{T}(vec::Vector{T}) = f::Symbol -> map(d -> getfield(d, f), vec)

function sumOverGen{T}(vec::Vector{T})
  pluck = pluckGen(vec)
  field::Symbol -> @> field pluck sum
end


immutable AnovaInfo
  groups_info::DataFrame
  results_info::DataFrame
end

function AnovaInfo(datagroups::Vector{DataGroup},
                   df_within::Int64, ss_within::Float64, ms_within::Float64,
                   df_between::Int64, ss_between::Float64, ms_between::Float64,
                   df_total::Int64, ss_total::Float64,
                   fStat::Float64)
  pluck = pluckGen(datagroups)

  groups_info = DataFrame(label=pluck(:label),
                         n=pluck(:sample_size),
                         mean=pluck(:sample_mean),
                         stdError=pluck(:sample_stde))

  pv::Float64 = ccdf(FDist(df_between, df_within), fStat)
  results_info = DataFrame(source=["Within Group", "Between Group", "Total"],
                          df=[df_within, df_between, df_total],
                          sum_squares=[ss_within, ss_between, ss_total],
                          FStat=@data([fStat, NA, NA]),
                          PValue=@data([pv, NA, NA])
                          )
  AnovaInfo(groups_info, results_info)
end

Base.show(io::IO, ai::AnovaInfo) = begin
  print(io, "Groups Info\n")
  print(io, ai.groups_info)
  print(io, "\n")
  print(io, "Results Info\n")
  print(io, ai.results_info)
end

function calcanova(datagroups::AbstractVector{DataGroup})
  pluck = pluckGen(datagroups)
  sumOver = sumOverGen(datagroups)

  groupCount::Int64 = length(datagroups)
  dataCount::Int64 = @> :sample_size sumOver

  ss_within::Float64 = @> :sum_squares sumOver
  df_within::Int64 = dataCount - groupCount
  ms_within::Float64 = ss_within/df_within

  dataMean::Float64 = (@> :weighted_mean sumOver)/dataCount
  ss_between::Float64 = reduce(0., datagroups) do acc, d
    acc + d.sample_size*(d.sample_mean - dataMean)^2
  end
  df_between::Int64 = groupCount - 1
  ms_between::Float64 = ss_between/df_between

  ss_total::Float64 = reduce(0., datagroups) do acc, d
    acc + sum((d.data - dataMean).^2)
  end
  df_total::Int64 = dataCount - 1
  msTotal::Float64 = ss_total/df_total

  fs::Float64 = ms_between/ms_within

  AnovaInfo(datagroups,
            df_within, ss_within, ms_within,
            df_between, ss_between, ms_between,
            df_total, ss_total,
            fs)
end

calcanova(datagroups...) = calcanova(DataGroup[d for d in datagroups])
