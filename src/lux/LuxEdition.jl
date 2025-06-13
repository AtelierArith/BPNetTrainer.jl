module LuxEdition

using Random
using Dates

using LuxCUDA: CUDA
using Lux
using MLUtils: MLUtils
using Optimisers
using Setfield: @set!
using MPI: MPI
using NCCL: NCCL # Enables distributed training in Lux. NCCL is needed for CUDA GPUs
import Zygote # For AutoZygote

using ..BPNetTrainer: BPDataMemory, BPDataset, make_train_and_test_jld2

include("model.jl")
include("training.jl")


include("ddp_training.jl")

end
