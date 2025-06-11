@testitem "FluxBPNet Model Construction" begin
    using BPNetTrainer
    using BPNetTrainer.FluxEdition: FluxBPNet, make_dense_chain
    using Flux
    using TOML

    # Test make_dense_chain function
    inputdim = 10
    hidden_dims = [15, 15]
    activations = [Flux.tanh, Flux.tanh]

    # Test regular dense chain
    chain = make_dense_chain(inputdim, hidden_dims, activations, false)
    @test chain isa Flux.Chain
    @test length(chain.layers) == 3  # input->hidden1, hidden1->hidden2, hidden2->output

    # Test that output layer has no activation
    @test chain.layers[end] isa Flux.Dense
    @test chain.layers[end].σ === identity

    # Test input/output dimensions
    test_input = randn(Float32, inputdim, 1)
    output = chain(test_input)
    @test size(output) == (1, 1)  # Single output neuron

    # Test ResNet version
    chain_resnet = make_dense_chain(15, [15, 15], activations, true)
    @test chain_resnet isa Flux.Chain
    # First layer should be Parallel when input dim equals hidden dim
    @test chain_resnet.layers[1] isa Flux.Parallel
end

@testitem "FluxBPNet Configuration from TOML" begin
    using BPNetTrainer
    using BPNetTrainer.FluxEdition: FluxBPNet
    using BPNetTrainer: FingerPrintParams
    using Flux

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

    model = FluxBPNet(toml_config, fingerprint_params)

    @test model isa FluxBPNet
    @test length(model.chains) == 2  # Two atom types
    @test length(model.chains[1]) == 1  # numbasiskinds = 1
    @test length(model.chains[2]) == 1

    # Test that chains have correct architecture
    ti_chain = model.chains[1][1]
    o_chain = model.chains[2][1]

    @test ti_chain isa Flux.Chain
    @test o_chain isa Flux.Chain

    # Test input dimensions match fingerprint parameters
    test_ti_input = randn(Float32, 20, 5)  # Ti fingerprint dim = 20
    test_o_input = randn(Float32, 18, 3)   # O fingerprint dim = 18

    ti_output = ti_chain(test_ti_input)
    o_output = o_chain(test_o_input)

    @test size(ti_output, 1) == 1  # Single output per atom
    @test size(o_output, 1) == 1
end

@testitem "Model Activation Functions" begin
    using BPNetTrainer.FluxEdition: make_dense_chain
    using Flux

    # Test different activation functions
    inputdim = 5
    hidden_dims = [8]

    # Test with tanh
    activations_tanh = [Flux.tanh]
    chain_tanh = make_dense_chain(inputdim, hidden_dims, activations_tanh)
    @test chain_tanh.layers[1].σ === Flux.tanh

    # Test with relu
    activations_relu = [Flux.relu]
    chain_relu = make_dense_chain(inputdim, hidden_dims, activations_relu)
    @test chain_relu.layers[1].σ === Flux.relu

    # Test with mixed activations
    hidden_dims_multi = [8, 6]
    activations_mixed = [Flux.tanh, Flux.sigmoid]
    chain_mixed = make_dense_chain(inputdim, hidden_dims_multi, activations_mixed)
    @test chain_mixed.layers[1].σ === Flux.tanh
    @test chain_mixed.layers[2].σ === Flux.sigmoid
    @test chain_mixed.layers[3].σ === identity  # Output layer should have no activation
end

@testitem "FluxBPNet Multi-Basis Support" begin
    using BPNetTrainer
    using BPNetTrainer.FluxEdition: FluxBPNet
    using BPNetTrainer: FingerPrintParams

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

    model = FluxBPNet(toml_config, fingerprint_params)

    @test length(model.chains) == 1  # One atom type
    @test length(model.chains[1]) == 2  # Two basis kinds

    # Each chain should handle its respective input dimension
    chain1 = model.chains[1][1]
    chain2 = model.chains[1][2]

    test_input1 = randn(Float32, 15, 3)  # First basis
    test_input2 = randn(Float32, 12, 3)  # Second basis

    output1 = chain1(test_input1)
    output2 = chain2(test_input2)

    @test size(output1) == (1, 3)
    @test size(output2) == (1, 3)
end