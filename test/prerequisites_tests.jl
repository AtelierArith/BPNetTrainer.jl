@testitem "Dataset Download Functionality" begin
    using BPNetTrainer
    using BPNetTrainer: download_dataset, DATASET_ROOT
    using Test

    # Note: This test will actually download data on first run
    # Subsequent runs should detect existing files

    # Test download function doesn't error
    @test begin
        download_dataset(); true
    end


    # Test that dataset root directory exists
    @test isdir(DATASET_ROOT[])

    # Test that expected files exist after download
    bz2name = "aenet-example-02-TiO2-Chebyshev.tar.bz2"
    bz2path = joinpath(DATASET_ROOT[], bz2name)

    tarname = first(splitext(bz2name))
    tarpath = joinpath(DATASET_ROOT[], tarname)

    extractpath = joinpath(DATASET_ROOT[], "extracted_files")

    # Files should exist after successful download
    @test isfile(bz2path)
    @test isfile(tarpath)
    @test isdir(extractpath)

    # Test that example data directory exists
    example_dir = joinpath(extractpath, "aenet-example-02-TiO2-Chebyshev", "TiO2-xsf")
    @test isdir(example_dir)

    # Test that there are XSF files in the example directory
    xsf_files = filter(readdir(example_dir)) do x
        last(splitext(x)) == ".xsf"
    end
    @test length(xsf_files) > 0
end

@testitem "DataGenerator Construction" begin
    using BPNetTrainer
    using BPNetTrainer: DataGenerator, FingerPrint

    # Test basic DataGenerator construction
    types = ["Ti", "O"]
    g = DataGenerator(types)

    @test g isa DataGenerator
    @test g.types == types
    @test g.numkinds == length(types)
    @test g.numfiles == 0
    @test length(g.fingerprints) == length(types)
    @test g.isgenerated == false

    # Test with custom parameters
    isolated_energies = [-1.0, -2.0]
    outputname = "custom_test.train"
    fingerprint_names = ["Ti_custom.fingerprint.stp", "O_custom.fingerprint.stp"]

    g_custom = DataGenerator(
        types;
        isolated_energies=isolated_energies,
        outputname=outputname,
        fingerprint_names=fingerprint_names
    )

    @test g_custom.isolated_energies == isolated_energies
    @test g_custom.outputname == outputname
    @test g_custom.fingerprint_names == fingerprint_names

    # Test default naming
    @test g.outputname == "TiO.train"
    @test g.fingerprint_names == ["Ti.fingerprint.stp", "O.fingerprint.stp"]
    @test all(g.isolated_energies .== 0.0)
end

@testitem "DataGenerator Data Management" begin
    using BPNetTrainer
    using BPNetTrainer: DataGenerator, adddata!, set_numfiles!

    types = ["Ti", "O"]
    g = DataGenerator(types)

    # Test adding single file
    test_file = "test_structure.xsf"
    adddata!(g, test_file)

    @test g.numfiles == 1
    @test test_file in g.filenames

    # Test adding multiple files
    test_files = ["struct1.xsf", "struct2.xsf", "struct3.xsf"]
    adddata!(g, test_files)

    @test g.numfiles == 4  # 1 + 3
    @test all(f in g.filenames for f in test_files)

    # Test setting number of files
    set_numfiles!(g, 100)
    @test g.numfiles == 100
end

@testitem "DataGenerator FingerPrint Integration" begin
    using BPNetTrainer
    using BPNetTrainer: DataGenerator, FingerPrint

    types = ["Ti", "O"]
    g = DataGenerator(types)

    # Create fingerprints for each atom type
    fp_ti = FingerPrint(
        "Ti", types;
        basistype="Chebyshev",
        radial_Rc=4.0,
        radial_N=10,
        angular_Rc=3.5,
        angular_N=8
    )

    fp_o = FingerPrint(
        "O", types;
        basistype="Chebyshev",
        radial_Rc=4.0,
        radial_N=10,
        angular_Rc=3.5,
        angular_N=8
    )

    # Test error on mismatched environment types
    wrong_types = ["Al", "Si"]
    fp_wrong = FingerPrint(
        "Al", wrong_types;
        basistype="Chebyshev",
        radial_Rc=4.0,
        radial_N=10,
        angular_Rc=3.5,
        angular_N=8
    )

    @test_throws AssertionError push!(g, fp_wrong)
end

@testitem "Generate Input File Creation" begin
    using BPNetTrainer
    using BPNetTrainer: DataGenerator, make_generatein, DATASET_ROOT, adddata!
    using TOML

    types = ["Ti", "O"]
    isolated_energies = [-1.5, -2.0]
    g = DataGenerator(types; isolated_energies=isolated_energies)

    # Add some test data
    test_files = ["struct1.xsf", "struct2.xsf"]
    adddata!(g, test_files)

    # Create generate.in file
    filename = make_generatein(g)

    @test filename == "generate.in"

    # Test that file was created
    filepath = joinpath(DATASET_ROOT[], filename)
    @test isfile(filepath)

    # Read and verify contents
    content = read(filepath, String)

    @test occursin("OUTPUT $(g.outputname)", content)
    @test occursin("TYPES", content)
    @test occursin("$(g.numkinds)", content)
    @test occursin("Ti $(isolated_energies[1])", content)
    @test occursin("O $(isolated_energies[2])", content)
    @test occursin("SETUPS", content)
    @test occursin("FILES", content)
    @test occursin("$(g.numfiles)", content)

    for file in test_files
        @test occursin(file, content)
    end

    # Clean up
    rm(filepath)
end

@testitem "FingerPrint File Generation" begin
    using BPNetTrainer
    using BPNetTrainer: DataGenerator, make_fingerprintfile, FingerPrint, DATASET_ROOT

    types = ["Ti", "O"]
    g = DataGenerator(types)

    # Create and add fingerprints
    fp_ti = FingerPrint(
        "Ti", types;
        basistype="Chebyshev",
        radial_Rc=8.0,
        radial_N=16,
        angular_Rc=6.5,
        angular_N=6
    )

    fp_o = FingerPrint(
        "O", types;
        basistype="Chebyshev",
        radial_Rc=8.0,
        radial_N=16,
        angular_Rc=6.5,
        angular_N=6
    )

    push!(g, fp_ti)
    push!(g, fp_o)

    # Generate fingerprint files
    make_fingerprintfile(g)

    # Test that files were created
    for (i, name) in enumerate(g.fingerprint_names)
        filepath = joinpath(DATASET_ROOT[], name)
        @test isfile(filepath)

        # Read and verify basic structure
        content = read(filepath, String)
        @test occursin("DESCR", content)
        @test occursin("END DESCR", content)
        @test occursin("ATOM $(types[i])", content)
        @test occursin("ENV $(length(types))", content)
        @test occursin("RMIN", content)
        @test occursin("BASIS type=Chebyshev", content)
        @test occursin("radial_Rc", content)
        @test occursin("radial_N", content)
        @test occursin("angular_Rc", content)
        @test occursin("angular_N", content)

        # Clean up
        rm(filepath)
    end
end
