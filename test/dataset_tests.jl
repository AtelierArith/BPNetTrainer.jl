#=
@testitem "Dataset Construction and Basic Operations" begin

    using BPNetTrainer
    using BPNetTrainer: BPDataset, BPDataMemory, make_train_and_test_jld2
    using TOML
    using JLD2
    using Random

    # Set up test environment
    Random.seed!(42)

    # Test BPDataset construction from TOML config
    tomlpath = joinpath(pkgdir(BPNetTrainer), "configs", "test_input.toml")
    @test isfile(tomlpath)

    # Download and generate test data (should not fail)
    BPNetTrainer.download_dataset()
    BPNetTrainer.generate_example_dataset()

    # Test dataset creation
    bpdata, toml_config = BPDataset(tomlpath)

    @test bpdata isa BPDataset
    @test toml_config isa Dict
    @test length(bpdata) > 0
    @test bpdata.num_of_types == 2  # Ti and O atoms
    @test bpdata.type_names == ["Ti", "O"]
    @test haskey(bpdata.fingerprints, :Ti)
    @test haskey(bpdata.fingerprints, :O)

    # Test input dimensions for each atom type
    @test BPNetTrainer.get_inputdim(bpdata, "Ti") > 0
    @test BPNetTrainer.get_inputdim(bpdata, "O") > 0
    @test BPNetTrainer.get_inputdim(bpdata, :Ti) == BPNetTrainer.get_inputdim(bpdata, "Ti")

    # Test train/test split functionality
    ratio = toml_config["testratio"]
    filename_train = toml_config["filename_train"]
    filename_test = toml_config["filename_test"]

    make_train_and_test_jld2(bpdata, filename_train, filename_test; ratio)

    # Verify files were created
    train_path = joinpath(BPNetTrainer.DATASET_ROOT[], filename_train)
    test_path = joinpath(BPNetTrainer.DATASET_ROOT[], filename_test)
    @test isfile(train_path)
    @test isfile(test_path)
end
=#

@testitem "BPDataMemory Operations" begin
    using BPNetTrainer
    using BPNetTrainer: BPDataset, BPDataMemory, make_train_and_test_jld2
    using Random

    Random.seed!(42)

    # Set up test data
    BPNetTrainer.download_dataset()
    BPNetTrainer.generate_example_dataset()

    tomlpath = joinpath(pkgdir(BPNetTrainer), "configs", "test_input.toml")
    bpdata, toml_config = BPDataset(tomlpath)

    # Create train/test split
    ratio = toml_config["testratio"]
    filename_train = toml_config["filename_train"]
    filename_test = toml_config["filename_test"]
    make_train_and_test_jld2(bpdata, filename_train, filename_test; ratio)

    # Test BPDataMemory creation
    traindata = BPDataMemory(bpdata, filename_train)
    testdata = BPDataMemory(bpdata, filename_test)

    @test traindata isa BPDataMemory
    @test testdata isa BPDataMemory
    @test length(traindata) > 0
    @test length(testdata) > 0
    @test length(traindata) + length(testdata) == length(bpdata)

    # Test data access
    x, y, num, totalnumatom = traindata[1]
    @test x isa Vector  # Should be vector of named tuples with data and labels
    @test y isa Matrix  # Energy batch
    @test size(y, 1) == 1  # Single energy value
    @test size(y, 2) == 1  # Single structure
    @test num == 1
    @test totalnumatom > 0

    # Test batch access
    batch_indices = 1:min(5, length(traindata))
    x_batch, y_batch, num, totalnumatom = traindata[batch_indices]
    @test size(y_batch, 2) == length(batch_indices)
    @test length(x_batch) == bpdata.num_of_types  # One entry per atom type
    @test num == length(batch_indices)
    @test totalnumatom > 0

    # Test that each element in x_batch has proper structure
    for atom_data in x_batch
        @test haskey(atom_data, :data)
        @test haskey(atom_data, :labels)
        @test atom_data.data isa Vector{Matrix{Float64}}  # Coefficients for each basis kind
        @test atom_data.labels isa Matrix{Float64}  # Structure indices
    end
end
