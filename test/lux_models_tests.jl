@testitem "LuxBPNet Model Construction" begin
    using BPNetTrainer
    using BPNetTrainer.LuxEdition: LuxBPNet, make_dense_chain, LayerAcceptsInputWithDataAndLabels
    using Lux
    using LuxCore
    using Random
    
    # Test make_dense_chain function
    inputdim = 10
    hidden_dims = [15, 15]
    activations = [Lux.tanh, Lux.tanh]
    
    # Test regular dense chain
    chain = make_dense_chain(inputdim, hidden_dims, activations, false)
    @test chain isa Lux.Chain
    @test length(chain.layers) == 3  # input->hidden1, hidden1->hidden2, hidden2->output
    
    # Test that output layer has no activation (identity)
    @test chain.layers[end] isa Lux.Dense
    @test chain.layers[end].activation === identity
    
    # Test ResNet version
    chain_resnet = make_dense_chain(15, [15, 15], activations, true)
    @test chain_resnet isa Lux.Chain
    # First layer should be Parallel when input dim equals hidden dim
    @test chain_resnet.layers[1] isa Lux.Parallel
    
    # Test forward pass
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, chain)
    test_input = randn(Float32, inputdim, 1)
    output, _ = Lux.apply(chain, test_input, ps, st)
    @test size(output) == (1, 1)  # Single output neuron
end

@testitem "LayerAcceptsInputWithDataAndLabels" begin
    using BPNetTrainer.LuxEdition: LayerAcceptsInputWithDataAndLabels
    using Lux
    using Random
    
    # Create a simple dense layer wrapped in LayerAcceptsInputWithDataAndLabels
    inner_layer = Lux.Dense(5, 3)
    wrapper = LayerAcceptsInputWithDataAndLabels(inner_layer)
    
    @test wrapper isa LayerAcceptsInputWithDataAndLabels
    @test wrapper isa LuxCore.AbstractLuxContainerLayer
    
    # Test forward pass
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, wrapper)
    
    # Create input in expected format (named tuple with data and labels)
    test_input = (
        data = (randn(Float32, 5, 2),),  # Tuple of data matrices
        labels = randn(Float32, 3, 2)    # Labels matrix
    )
    
    output, new_st = Lux.apply(wrapper, test_input, ps, st)
    
    @test output isa AbstractArray
    @test size(output) == (3, 2)  # Should match labels dimensions
    @test haskey(new_st, :layer)
end

@testitem "LuxBPNet Configuration from TOML" begin
    using BPNetTrainer
    using BPNetTrainer.LuxEdition: LuxBPNet
    using BPNetTrainer: FingerPrintParams
    using Lux
    using LuxCore
    using Random
    
    # Create mock TOML configuration
    toml_config = Dict{String, Any}(
        "atomtypes" => ["Ti", "O"],
        "numbasiskinds" => 1,
        "Ti" => Dict("layers" => [15, 15], "activations" => ["tanh", "tanh"]),
        "O" => Dict("layers" => [10, 10], "activations" => ["relu", "relu"])
    )
    
    # Create mock fingerprint parameters
    fingerprint_params = [
        [FingerPrintParams("Chebyshev", 1, 20, [4.0, 10.0, 3.5, 8.0], 1, 20)],  # Ti
        [FingerPrintParams("Chebyshev", 1, 18, [4.0, 8.0, 3.0, 6.0], 1, 18)]    # O
    ]
    
    model = LuxBPNet(toml_config, fingerprint_params)
    
    @test model isa LuxBPNet
    @test model isa LuxCore.AbstractLuxContainerLayer
    @test model.layer isa Lux.Parallel
    
    # Test model initialization
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, model)
    
    @test ps isa NamedTuple
    @test st isa NamedTuple
    @test haskey(ps, :layer)
    @test haskey(st, :layer)
end

@testitem "LuxBPNet Forward Pass" begin
    using BPNetTrainer
    using BPNetTrainer.LuxEdition: LuxBPNet
    using BPNetTrainer: FingerPrintParams
    using Lux
    using Random
    
    # Create a simple test model
    toml_config = Dict{String, Any}(
        "atomtypes" => ["Ti", "O"],
        "numbasiskinds" => 1,
        "Ti" => Dict("layers" => [5, 5], "activations" => ["tanh", "tanh"]),
        "O" => Dict("layers" => [5, 5], "activations" => ["tanh", "tanh"])
    )
    
    fingerprint_params = [
        [FingerPrintParams("Chebyshev", 1, 10, [4.0, 10.0, 3.5, 8.0], 1, 10)],  # Ti
        [FingerPrintParams("Chebyshev", 1, 8, [4.0, 8.0, 3.0, 6.0], 1, 8)]      # O
    ]
    
    model = LuxBPNet(toml_config, fingerprint_params)
    
    # Initialize model
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, model)
    
    # Create test input data (mimicking BPDataMemory output format)
    # Format: tuple of tuples with (data_tuple, labels)
    test_input = (
        ((randn(Float32, 10, 3),), randn(Float32, 2, 3)),  # Ti atoms
        ((randn(Float32, 8, 2),), randn(Float32, 2, 2))    # O atoms
    )
    
    # Test forward pass
    output, new_st = Lux.apply(model, test_input, ps, st)
    
    @test output isa AbstractArray
    @test size(output, 1) == 1  # Single energy output
    @test size(output, 2) == 2  # Batch size
    @test new_st isa NamedTuple
    @test haskey(new_st, :layer)
end

@testitem "LuxBPNet Multi-Basis Support" begin
    using BPNetTrainer
    using BPNetTrainer.LuxEdition: LuxBPNet
    using BPNetTrainer: FingerPrintParams
    using Lux
    using Random
    
    # Test with multiple basis kinds
    toml_config = Dict{String, Any}(
        "atomtypes" => ["Ti"],
        "numbasiskinds" => 2,  # Multiple basis functions
        "Ti" => Dict("layers" => [10, 10], "activations" => ["tanh", "tanh"])
    )
    
    fingerprint_params = [
        [
            FingerPrintParams("Chebyshev", 2, 15, [4.0, 10.0, 3.5, 8.0], 1, 15),
            FingerPrintParams("Spline", 2, 12, [3.0, 8.0, 2.5, 6.0], 16, 27)
        ]
    ]
    
    model = LuxBPNet(toml_config, fingerprint_params)
    
    # Initialize model
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, model)
    
    @test model.layer isa Lux.Parallel
    
    # Create test input with multiple basis
    test_input = (
        ((randn(Float32, 15, 3), randn(Float32, 12, 3)), randn(Float32, 2, 3)),  # Ti with 2 basis
    )
    
    output, new_st = Lux.apply(model, test_input, ps, st)
    
    @test output isa AbstractArray
    @test size(output, 1) == 1
    @test size(output, 2) == 2
end

@testitem "OnlyFollowsLossFn Wrapper" begin
    using BPNetTrainer.LuxEdition: OnlyFollowsLossFn
    using Lux
    
    # Test OnlyFollowsLossFn wrapper
    mse_loss = Lux.MSELoss()
    wrapped_loss = OnlyFollowsLossFn(mse_loss)
    
    @test wrapped_loss isa OnlyFollowsLossFn
    @test wrapped_loss isa Lux.AbstractLossFunction
    
    # Test that it extracts scalar from loss function result
    y_pred = randn(Float32, 1, 5)
    y_true = randn(Float32, 1, 5)
    
    loss_value = wrapped_loss(y_pred, y_true)
    @test loss_value isa Number  # Should be a scalar
    
    # Compare with original loss (which might return array)
    original_loss = mse_loss(y_pred, y_true)
    if original_loss isa AbstractArray
        @test loss_value ≈ only(original_loss)
    else
        @test loss_value ≈ original_loss
    end
end

@testitem "Lux Model Activation Functions" begin
    using BPNetTrainer.LuxEdition: make_dense_chain
    using Lux
    using Random
    
    # Test different activation functions
    inputdim = 5
    hidden_dims = [8]
    rng = Xoshiro(42)
    
    # Test with tanh
    activations_tanh = [Lux.tanh]
    chain_tanh = make_dense_chain(inputdim, hidden_dims, activations_tanh)
    ps, st = Lux.setup(rng, chain_tanh)
    @test chain_tanh.layers[1].activation === Lux.tanh
    
    # Test with relu
    activations_relu = [Lux.relu]
    chain_relu = make_dense_chain(inputdim, hidden_dims, activations_relu)
    ps, st = Lux.setup(rng, chain_relu)
    @test chain_relu.layers[1].activation === Lux.relu
    
    # Test with mixed activations
    hidden_dims_multi = [8, 6]
    activations_mixed = [Lux.tanh, Lux.sigmoid]
    chain_mixed = make_dense_chain(inputdim, hidden_dims_multi, activations_mixed)
    ps, st = Lux.setup(rng, chain_mixed)
    @test chain_mixed.layers[1].activation === Lux.tanh
    @test chain_mixed.layers[2].activation === Lux.sigmoid
    @test chain_mixed.layers[3].activation === identity  # Output layer should have no activation
end

@testitem "Lux Model Parameter Types" begin
    using BPNetTrainer.LuxEdition: LuxBPNet
    using BPNetTrainer: FingerPrintParams
    using Lux
    using Random
    
    # Create model with specific precision
    toml_config = Dict{String, Any}(
        "atomtypes" => ["Ti"],
        "numbasiskinds" => 1,
        "Ti" => Dict("layers" => [8, 8], "activations" => ["tanh", "tanh"])
    )
    
    fingerprint_params = [
        [FingerPrintParams("Chebyshev", 1, 10, [4.0, 10.0, 3.5, 8.0], 1, 10)]
    ]
    
    model = LuxBPNet(toml_config, fingerprint_params)
    
    # Test with Float32
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, model)
    
    # Check that parameters are properly typed
    @test ps isa NamedTuple
    @test st isa NamedTuple
    
    # Test forward pass maintains type consistency
    test_input = (((randn(Float32, 10, 2),), randn(Float32, 1, 2)),)
    output, _ = Lux.apply(model, test_input, ps, st)
    
    @test eltype(output) <: AbstractFloat
end