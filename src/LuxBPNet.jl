module LuxBPNet

using Downloads
using TOML

using BPNET_jll: BPNET_jll
using Tar: Tar
using CodecBzip2: CodecBzip2
using TranscodingStreams: TranscodingStreams
using Scratch: @get_scratch!

const DATASET_DIR = Ref{String}()

include("data_downloader.jl")
include("fingerprints.jl")
include("data_generator.jl")
include("dataset.jl")

function __init__()
    global DATASET_DIR
    DATASET_DIR[] = @get_scratch!("dataset")
end

end # module LuxBPNet
