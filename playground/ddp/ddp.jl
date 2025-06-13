using Pkg
using Revise
Pkg.activate(dirname(dirname(@__DIR__)))
#Pkg.instantiate()

using Random
using Lux
using MLUtils
using Optimisers
using LuxCUDA

using BPNetTrainer
using BPNetTrainer: BPDataset
using BPNetTrainer: download_dataset, generate_example_dataset

# Distributed Training: NCCL for NVIDIA GPUs and MPI for anything else
const distributed_backend = try
    if gpu_device() isa CUDADevice
        DistributedUtils.initialize(NCCLBackend)
        DistributedUtils.get_distributed_backend(NCCLBackend)
    else
        DistributedUtils.initialize(MPIBackend)
        DistributedUtils.get_distributed_backend(MPIBackend)
    end
catch err
    @error "Could not initialize distributed training. Error: $err"
    nothing
end

@assert !isnothing(distributed_backend) "Distributed backend must be initialized before training."

#!download_dataset()
#generate_example_dataset(500)

tomlpath = joinpath(pkgdir(BPNetTrainer), "configs", "test_input.toml")

@time BPNetTrainer.LuxEdition.ddptraining(distributed_backend, tomlpath)

