module LuxBPNet

using Downloads
using TOML
using Random: shuffle
using SparseArrays: sparse

using BPNET_jll: BPNET_jll
using Tar: Tar
using CodecBzip2: CodecBzip2
using TranscodingStreams: TranscodingStreams
using Scratch: @get_scratch!
using JLD2: JLD2, jldopen

const DATASET_ROOT = Ref{String}()

function switchdatasetdir!(dir::AbstractString)
    global DATASET_ROOT
    DATASET_ROOT[] = string(dir)
end

include("fingerprints.jl")
include("prerequisites/generator.jl")
include("prerequisites/downloader.jl")
include("dataset/bpdataset.jl")
include("dataset/jld2writer.jl")
include("dataset/bpdatamemory.jl")

function __init__()
    global DATASET_ROOT
    DATASET_ROOT[] = @get_scratch!("dataset")
end

end # module LuxBPNet
