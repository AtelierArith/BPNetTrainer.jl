@testitem "LuxBPNet Configuration from TOML" begin
    using BPNetTrainer
    using BPNetTrainer.LuxEdition: LuxBPNet
    using BPNetTrainer: FingerPrintParams
    using Lux
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
    @test model isa Lux.AbstractLuxContainerLayer
    @test model.layer isa Lux.Parallel

    # Test model initialization
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, model)

    @test ps isa NamedTuple
    @test st isa NamedTuple
    @test haskey(ps, :layer)
    @test haskey(st, :layer)
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
