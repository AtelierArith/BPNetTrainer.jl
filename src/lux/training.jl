
struct OnlyFollowsLossFn{F} <: Lux.AbstractLossFunction
    fn_internal::F
end

function (fn::OnlyFollowsLossFn)(ŷ, y)
    only(fn.fn_internal(ŷ, y))
end

"""
    training(bpdata, toml)

Train a neural network using the Lux.jl framework.

This is the main training function for the Lux edition of BPNetTrainer. It handles
the complete training pipeline including data preparation, model creation, 
optimization setup, and training loop execution.

# Arguments
- `bpdata::BPDataset`: Dataset containing training structures and fingerprints
- `toml::Dict`: Configuration dictionary (typically from TOML file)

# Required TOML Configuration
- `testratio::Float64`: Fraction of data for testing (e.g., 0.1 for 10%)
- `filename_train::String`: Name for training data JLD2 file
- `filename_test::String`: Name for test data JLD2 file  
- `numbatch::Int`: Batch size for training
- `nepoch::Int`: Number of training epochs
- `lr::Float64`: Learning rate for Adam optimizer

# Model Configuration (per atom type in toml)
```toml
[model.Ti]
layers = [100, 50, 25, 1]  # Hidden layer sizes + output
activations = ["tanh", "tanh", "tanh", "identity"]
```

# Process
1. Creates train/test split using `make_train_and_test_jld2`
2. Loads data into memory using `BPDataMemory`
3. Sets up data loaders with specified batch size
4. Creates `LuxBPNet` model from configuration
5. Initializes model parameters and optimizer state
6. Runs training loop with logging and GPU acceleration

# Output
- Prints training progress (epoch, train loss, test loss)
- Uses GPU acceleration if available
- Model parameters are updated in-place during training

# Example
```julia
using BPNetTrainer

# Load configuration and data
tomlpath = joinpath(pkgdir(BPNetTrainer), "configs", "test_input.toml")
bpdata, toml = BPDataset(tomlpath)

# Train the model
BPNetTrainer.LuxEdition.training(bpdata, toml)
```

# See also
- [`BPNetTrainer.FluxEdition.training!`](@ref): Flux version
- [`LuxBPNet`](@ref): Neural network model structure  
- [`BPDataMemory`](@ref): Memory-efficient data loading
"""
function training(bpdata, toml)
    device = gpu_device()

    ratio = toml["testratio"]
    filename_train = toml["filename_train"]
    filename_test = toml["filename_test"]
    make_train_and_test_jld2(bpdata, filename_train, filename_test; ratio)

    numbatch = toml["numbatch"]
    traindata = BPDataMemory(bpdata, filename_train)
    train_loader = MLUtils.DataLoader(traindata; batchsize = numbatch, shuffle = true)

    testdata = BPDataMemory(bpdata, filename_test)
    test_loader = MLUtils.DataLoader(testdata; batchsize = 1)

    model = LuxBPNet(toml, bpdata.fingerprint_parameters)

    rng = Xoshiro(1234)
    _ps, _st = Lux.setup(rng, model)
    ps = _ps |> device
    st = _st |> device
    tstate = Lux.Training.TrainState(model, ps, st, Optimisers.AdamW())
    lossfn = OnlyFollowsLossFn(Lux.MSELoss())
    nepoch = toml["nepoch"]
    for epoch = 1:nepoch
        @info epoch
        @info ("Training phase")
        st = Lux.trainmode(st)

        train_loss = 0.0
        train_sse = 0.0
        for (i, (x, y, num, totalnumatom)) in enumerate(train_loader)
            x_dev = [(Tuple(device(Lux.f32(e.data))), device(Lux.f32(e.labels))) for e in x]
            y_dev = y |> Lux.f32 |> device

            # ŷ, _ = Lux.apply(model, x_dev, ps, st)
            _, loss, _, tstate = Lux.Training.single_train_step!(
                AutoZygote(),
                lossfn,
                (x_dev, y_dev),
                tstate,
            )
            train_loss += cpu_device()(loss)
            train_sse += loss / totalnumatom^2
        end
        train_sse = train_sse / length(train_loader)
        train_rmse = sqrt(train_sse) / train_loader.data.E_scale
        @info ("train loss: ", train_loss / length(train_loader))
        @info ("train rmse: ", train_rmse, "[eV/atom]")

        @info ("Validation phase")
        st = Lux.testmode(st)

        test_loss = 0.0
        test_sse = 0.0
        for (i, (x, y, num, totalnumatom)) in enumerate(test_loader)
            x_dev = [(Tuple(device(Lux.f32(e.data))), device(Lux.f32(e.labels))) for e in x]
            y_dev = y |> Lux.f32 |> device

            ŷ, _ = Lux.apply(model, x_dev, ps, st)
            loss = lossfn(ŷ, y_dev)
            test_loss += cpu_device()(loss)
            test_sse += loss / totalnumatom^2
        end
        test_sse = test_sse / length(test_loader)
        test_rmse = sqrt(test_sse) / test_loader.data.E_scale
        @info ("test loss: ", test_loss / length(test_loader))
        @info ("test rmse: ", test_rmse / length(test_loader), "[eV/atom]")
    end
end
