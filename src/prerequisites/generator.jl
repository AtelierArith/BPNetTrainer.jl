mutable struct DataGenerator
    outputname::String
    numkinds::Int64
    types::Vector{String}
    fingerprints::Vector{FingerPrint}
    filenames::Vector{String}
    numfiles::Int64
    isolated_energies::Vector{Float64}
    fingerprint_names::Vector{String}
    isgenerated::Bool
end

function DataGenerator(types; kwargs...)
    numkinds = length(types)
    #types = Vector{String}(undef, numkinds)
    fingerprints = Vector{FingerPrint}(undef, numkinds)
    filenames = String[]
    if haskey(kwargs, :isolated_energies)
        isolated_energies = kwargs[:isolated_energies]
    else
        isolated_energies = zeros(numkinds)
    end

    if haskey(kwargs, :outputname)
        outputname = kwargs[:outputname]
    else
        outputname = prod(types) * ".train"
    end


    if haskey(kwargs, :fingerprint_names)
        fingerprint_names = kwargs[:fingerprint_names]
    else
        fingerprint_names = Vector{String}(undef, numkinds)
        for i = 1:numkinds
            fingerprint_names[i] = types[i] * ".fingerprint.stp"
        end
    end


    isgenerated = false
    return DataGenerator(
        outputname,
        numkinds,
        types,
        fingerprints,
        filenames,
        0,
        isolated_energies,
        fingerprint_names,
        isgenerated,
    )
end

function bpnet_jll_generatefn(filename)
    exe = BPNET_jll.generate()
    old = pwd()
    cd(DATASET_ROOT[])
    run(`$(exe) $(filename)`)
    cd(old)
    nothing
end

function make_descriptor(g::DataGenerator)
    d = DATASET_ROOT[]
    if isfile(joinpath(d, g.outputname))
        @warn "$(g.outputname) exists. This is removed"
        rm(joinpath(d, g.outputname))
    end

    filename = make_generatein(g)
    make_fingerprintfile(g)
    println(joinpath(d, filename))
    bpnet_jll_generatefn(joinpath(d, filename))
    outputfile = joinpath(d, g.outputname * ".ascii")
    g.isgenerated = true
    return outputfile
end

function make_fingerprintfile(g::DataGenerator)
    d = DATASET_ROOT[]
    for ifile = 1:g.numkinds
        filename = g.fingerprint_names[ifile]
        fingerprint = g.fingerprints[ifile]
        description = fingerprint.description
        fp = open(joinpath(d, filename), "w")
        println(fp, "DESCR")
        println(fp, description)
        println(fp, "END DESCR")
        println(fp, "")
        println(fp, "ATOM ", fingerprint.atomtype)
        println(fp, "")
        println(fp, "ENV ", fingerprint.nenv)
        for i = 1:fingerprint.nenv
            println(fp, fingerprint.envtypes[i])
        end
        println(fp, "")
        println(fp, "RMIN ", fingerprint.rc_min)
        println(fp, "")
        println(fp, "BASIS type=", fingerprint.sftype)
        print_fingerprintinfo(fp, fingerprint)
        close(fp)
    end
end

function print_fingerprintinfo(fp, fingerprint)
    if fingerprint.sftype == "Chebyshev"
        sfparam = fingerprint.sfparam
        radial_Rc = sfparam[1, 1]
        radial_N = Int64(sfparam[2, 1])
        angular_Rc = sfparam[3, 1]
        angular_N = Int64(sfparam[4, 1])

        print(fp, "radial_Rc = ", radial_Rc)
        print(fp, " radial_N = ", radial_N)
        print(fp, " angular_Rc = ", angular_Rc)
        print(fp, " angular_N = ", angular_N)
        println(fp, "\t")
    else
        error("fingerprint type $(fingerprint.sftype) is not supported")
    end
end

function make_generatein(g::DataGenerator; filename = "generate.in")
    d = DATASET_ROOT[]
    fp = open(joinpath(d, filename), "w")
    println(fp, "OUTPUT $(g.outputname)")
    println(fp, "\t")
    println(fp, "TYPES")
    println(fp, g.numkinds)
    for i = 1:g.numkinds
        println(fp, g.types[i], " ", g.isolated_energies[i], "  | eV")
    end
    println(fp, "\t")
    println(fp, "SETUPS")
    for i = 1:g.numkinds
        println(fp, g.types[i], " ", g.fingerprint_names[i])
    end
    println(fp, "\t")
    println(fp, "FILES")
    println(fp, g.numfiles)
    for i = 1:g.numfiles
        println(fp, g.filenames[i])
    end
    close(fp)
    return filename
end

function Base.push!(g::DataGenerator, f::FingerPrint)
    @assert g.types == f.envtypes "atomic type is wrong in fingerprints!"
    g.fingerprints[f.itype] = deepcopy(f)
end

function adddata!(g::DataGenerator, data::String)
    push!(g.filenames, data)
    g.numfiles += 1
end

function adddata!(g::DataGenerator, data::Vector{String})
    for datai in data
        adddata!(g, datai)
    end
end

function set_numfiles!(g::DataGenerator, numfiles)
    g.numfiles = numfiles
end

"""
    generate_example_dataset()

Generate processed training data from the downloaded TiO₂ example dataset.

This function creates fingerprint descriptors and processes the downloaded
XSF structure files into a format suitable for neural network training.
It sets up Chebyshev basis functions for both Ti and O atoms and processes
up to 5000 structures from the example dataset.

# Prerequisites
Must call [`download_dataset()`](@ref) first to obtain the raw XSF files.

# Process
1. Creates DataGenerator with Ti and O environment types
2. Sets up Chebyshev fingerprints for Ti atoms (radial_Rc=8.0, angular_Rc=6.5)
3. Sets up Chebyshev fingerprints for O atoms (same parameters)
4. Locates XSF files in the extracted dataset
5. Processes up to 5000 structures to create descriptor files

# Fingerprint Parameters
- **Radial functions**: Rc=8.0 Å, N=16 basis functions
- **Angular functions**: Rc=6.5 Å, N=6 basis functions
- **Basis type**: Chebyshev polynomials

# Output Files
Creates descriptor files in the dataset directory that can be used with:
- [`BPDataset(tomlpath)`](@ref) for loading training data
- Configuration files in `configs/` directory

# Example
```julia
using BPNetTrainer

# First download the raw data
download_dataset()

# Then process it for training
generate_example_dataset()

# Now ready to load for training
tomlpath = joinpath(pkgdir(BPNetTrainer), "configs", "test_input.toml")
bpdata, toml = BPDataset(tomlpath)
```

# See also
- [`download_dataset()`](@ref): Downloads the raw XSF files
- [`DataGenerator`](@ref): Core data processing functionality
- [`make_descriptor()`](@ref): Low-level descriptor generation
"""
function generate_example_dataset(numfiles = 5000)
    envtypes = ["Ti", "O"]
    g = DataGenerator(envtypes)

    atomtype = "Ti"
    f1 = FingerPrint(
        atomtype,
        envtypes;
        basistype = "Chebyshev",
        radial_Rc = 8.0,
        radial_N = 16,
        angular_Rc = 6.5,
        angular_N = 6,
    )
    #f1 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)

    push!(g, f1)

    atomtype = "O"
    f2 = FingerPrint(
        atomtype,
        envtypes;
        basistype = "Chebyshev",
        radial_Rc = 8.0,
        radial_N = 16,
        angular_Rc = 6.5,
        angular_N = 6,
    )
    #f2 = FingerPrint(atomtype, envtypes; basistype="Chebyshev", radial_Rc=8.0, radial_N=10, angular_Rc=6.5, angular_N=4)

    push!(g, f2)

    exampledir = joinpath(
        BPNetTrainer.DATASET_ROOT[],
        "extracted_files/aenet-example-02-TiO2-Chebyshev/TiO2-xsf",
    )

    xsf_files = filter(readdir(exampledir, join = true)) do x
        last(splitext(x)) == ".xsf"
    end
    adddata!(g, xsf_files)
    set_numfiles!(g, numfiles)
    make_descriptor(g)
end
