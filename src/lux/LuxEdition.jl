module LuxEdition

using Random

using Lux
using MLUtils: MLUtils
using Optimisers

using ..BPNetTrainer: BPDataMemory, make_train_and_test_jld2

include("model.jl")
include("training.jl")

end
