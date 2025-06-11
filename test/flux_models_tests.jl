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

@testitem "FluxBPNet Forward Pass" begin
    using BPNetTrainer
    using BPNetTrainer.FluxEdition: FluxBPNet, apply_model, apply_bpmultimodel
    using BPNetTrainer: FingerPrintParams
    using Flux
    
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
    
    model = FluxBPNet(toml_config, fingerprint_params)
    
    # Create test input data (mimicking BPDataMemory output format)
    # Format: vector of named tuples with :data and :labels
    test_input = [
        (data = [randn(Float32, 10, 3)], labels = randn(Float32, 2, 3)),  # Ti atoms
        (data = [randn(Float32, 8, 2)], labels = randn(Float32, 2, 2))    # O atoms
    ]
    
    # Test forward pass
    output = model(test_input)
    @test output isa AbstractArray
    @test size(output) == (1, 2)  # Should match the batch size in labels
    
    # Test apply_model function
    chain = model.chains[1][1]
    data = randn(Float32, 10, 3)
    result = apply_model(chain, data)
    @test size(result, 1) == 1
    @test size(result, 2) == 3
    
    # Test apply_bpmultimodel function
    chains = model.chains[1]
    xi = test_input[1]
    result = apply_bpmultimodel(chains, xi)
    @test result isa AbstractArray
end

@testitem "Flux Training State Setup" begin
    using BPNetTrainer.FluxEdition: set_state
    using Optimisers
    using Flux
    
    # Create a simple model for testing
    model = Flux.Dense(10, 1)
    θ = Flux.params(model)
    
    # Test AdamW optimizer setup
    inputdata_adamw = Dict("optimiser" => "AdamW")
    state_adamw = set_state(inputdata_adamw, θ)
    @test state_adamw isa Optimisers.OptimiserState
    
    # Test Adam optimizer setup
    inputdata_adam = Dict("optimiser" => "Adam")
    state_adam = set_state(inputdata_adam, θ)
    @test state_adam isa Optimisers.OptimiserState
    
    # Test unsupported optimizer
    inputdata_invalid = Dict("optimiser" => "SGD")
    @test_throws ErrorException set_state(inputdata_invalid, θ)
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