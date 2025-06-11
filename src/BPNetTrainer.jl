module BPNetTrainer

using Downloads
using TOML
using Random: shuffle
using SparseArrays: sparse

using BPNET_jll: BPNET_jll
using Tar: Tar
using CodecBzip2: CodecBzip2
using TranscodingStreams: TranscodingStreams
using Scratch: @get_scratch!
using JLD2: JLD2, jldopen, @save
using Flux

const DATASET_ROOT = Ref{String}()

"""
    switchdatasetdir!(dir::AbstractString)

Change the directory used for storing datasets.

This function updates the global `DATASET_ROOT` reference to point to the specified directory.
The directory will be used for downloading and storing datasets used by BPNetTrainer.

# Arguments
- `dir::AbstractString`: Path to the new dataset directory

# Example
```julia
using BPNetTrainer
switchdatasetdir!("/path/to/my/datasets")
```
"""
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
include("flux/FluxEdition.jl")
include("lux/LuxEdition.jl")

"""
    __init__()

Initialize the BPNetTrainer module.

This function is automatically called when the module is loaded. It sets up the default
dataset directory using Julia's Scratch.jl package to create a persistent scratch space
for storing datasets.

The scratch directory is typically located at:
- `~/.julia/scratchspaces/<uuid>/dataset/` on Unix systems
- Similar location on other platforms as determined by Scratch.jl
"""
function __init__()
    global DATASET_ROOT
    DATASET_ROOT[] = @get_scratch!("dataset")
end

end # module BPNetTrainer
