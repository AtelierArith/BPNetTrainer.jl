module FluxModelTraining

using Flux
using JLD2: @save

include("model.jl")
include("training.jl")

end
