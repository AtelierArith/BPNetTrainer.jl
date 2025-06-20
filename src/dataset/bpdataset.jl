"""
    BPDataset{keys,num_of_types,num_of_structs,Tfp,Td,num_kinds}

Main dataset structure for BPNet training data.

This structure holds all information needed for neural network training including
atomic configurations, energies, fingerprints, and normalization parameters.
It provides efficient access to training data and handles file I/O operations.

# Type Parameters
- `keys`: Named tuple keys for atom types (e.g., `(:Ti, :O)`)
- `num_of_types`: Number of different atom types
- `num_of_structs`: Number of atomic structures in the dataset
- `Tfp`: Type of fingerprint data
- `Td`: Type of data file handle
- `num_kinds`: Number of fingerprint basis kinds

# Fields
- `filename::String`: Path to the main data file
- `num_of_types::Int64`: Number of atom types in the system
- `num_of_structs::Int64`: Total number of atomic structures
- `type_names::Vector{String}`: Names of atom types (e.g., ["Ti", "O"])
- `E_atom::Vector{Float64}`: Atomic reference energies
- `normalized::Bool`: Whether energies are normalized
- `E_scale::Float64`: Energy scaling factor
- `E_shift::Float64`: Energy shift value
- `natomtot::Int64`: Total number of atoms across all structures
- `E_avg::Float64`: Average energy
- `E_min::Float64`: Minimum energy in dataset
- `E_max::Float64`: Maximum energy in dataset
- `has_setups::Bool`: Whether fingerprint setups are available
- `fp::Tfp`: File pointer for data access
- `fingerprints::NamedTuple`: Fingerprint definitions for each atom type
- `headerposision::Int64`: Position of header in data file
- `datafilenames::Vector{String}`: List of data files
- `fileused::Vector{Bool}`: Track which files have been used
- `datafile::Td`: Data file handle
- `fingerprint_parameters::Vector{Vector{FingerPrintParams}}`: Fingerprint parameters

# See also
- [`BPDataset(tomlpath::AbstractString)`](@ref): Main constructor
- [`BPDataMemory`](@ref): Memory-efficient version for training
"""
struct BPDataset{keys,num_of_types,num_of_structs,Tfp,Td,num_kinds}
    filename::String
    num_of_types::Int64
    num_of_structs::Int64
    type_names::Vector{String}
    E_atom::Vector{Float64}
    normalized::Bool
    E_scale::Float64
    E_shift::Float64
    natomtot::Int64
    E_avg::Float64
    E_min::Float64
    E_max::Float64
    has_setups::Bool
    fp::Tfp
    fingerprints::NamedTuple{keys,NTuple{num_of_types,FingerPrint}}
    #fingerprints::Vector{FingerPrint}
    headerposision::Int64
    datafilenames::Vector{String}
    fileused::Vector{Bool}
    datafile::Td
    fingerprint_parameters::Vector{Vector{FingerPrintParams}}
end

"""
    get_inputdim(dataset::BPDataset, name::Union{String,Symbol})

Get the input dimension for a specific atom type.

Returns the number of symmetry functions (fingerprint dimensions) for the
specified atom type, which corresponds to the input size for the neural network.

# Arguments
- `dataset::BPDataset`: The dataset containing fingerprint information
- `name::Union{String,Symbol}`: Name of the atom type (e.g., "Ti" or :Ti)

# Returns
- `Int`: Number of fingerprint dimensions for the specified atom type

# Example
```julia
inputdim_Ti = get_inputdim(dataset, "Ti")
inputdim_O = get_inputdim(dataset, :O)
```
"""
function get_inputdim(dataset::BPDataset, name::String)
    return get_inputdim(dataset, Symbol(name))
end

function get_inputdim(dataset::BPDataset, name::Symbol)
    return getfield(dataset.fingerprints, name).nsf
end

"""
    Base.length(dataset::BPDataset)

Return the number of atomic structures in the dataset.

# Arguments
- `dataset::BPDataset`: The dataset

# Returns
- `Int`: Number of structures available for training
"""
function Base.length(dataset::BPDataset)
    return dataset.num_of_structs
end

"""
    reload_data!(dataset::BPDataset)

Reset the file usage tracking to allow reusing all data.

This function marks all data files as unused, allowing the dataset to cycle
through all available data again. Useful for multiple training epochs.

# Arguments
- `dataset::BPDataset`: The dataset to reset

# Returns
- `nothing`
"""
function reload_data!(dataset::BPDataset)
    dataset.fileused .= 0
    return nothing
end

"""
    BPDataset(tomlpath::AbstractString)

Construct a BPDataset from a TOML configuration file.

This is the main constructor for BPDataset that reads training data and configuration
from a TOML file. It handles file parsing, fingerprint setup, and data validation.

# Arguments
- `tomlpath::AbstractString`: Path to the TOML configuration file

# Returns
- `BPDataset`: Fully configured dataset ready for training
- `Dict`: Parsed TOML configuration data

# Required TOML fields
- `trainfile`: Path to the main training data file
- `atomtypes`: Vector of atom type names (e.g., ["Ti", "O"])
- `maxenergy`: Maximum energy threshold

# Example TOML structure
```toml
trainfile = "train.data"
atomtypes = ["Ti", "O"]
maxenergy = 1000.0

[fingerprint.Ti]
basistype = "Chebyshev"
radial_Rc = 6.0
radial_N = 10
angular_Rc = 4.0
angular_N = 5
```

# Example usage
```julia
tomlpath = "configs/test_input.toml"
bpdata, toml = BPDataset(tomlpath)
```

# Throws
- `AssertionError`: If atom types in file don't match TOML configuration
- `SystemError`: If files cannot be opened or read
"""
function BPDataset(tomlpath::AbstractString)
    d = DATASET_ROOT[]
    data = TOML.parsefile(tomlpath)
    display(data)
    filename = data["trainfile"]
    fp = open(joinpath(d, filename), "r")
    num_of_types = parse(Int64, split(readline(fp))[1])
    num_of_structs = parse(Int64, split(readline(fp))[1])
    type_names = Vector{String}(undef, num_of_types)
    type_names .= collect(split(readline(fp)))
    @assert type_names == data["atomtypes"] "atomtypes should be $type_names"
    println(type_names)
    keys = Tuple(Symbol.(type_names))
    maxenergy = data["maxenergy"]



    E_atom = zeros(Float64, num_of_types)
    E_atom .= parse.(Float64, split(readline(fp)))

    u = split(readline(fp))[1]
    normalized = ifelse(u == "T", true, false)

    u = split(readline(fp))[1]
    E_scale = parse(Float64, u)
    u = split(readline(fp))[1]
    E_shift = parse(Float64, u)


    u = split(readline(fp))[1]
    natomtot = parse(Int64, u)
    E_avg, E_min, E_max = parse.(Float64, split(readline(fp)))
    u = split(readline(fp))[1]
    has_setups = ifelse(u == "T", true, false)


    fingerprints = Vector{FingerPrint}(undef, num_of_types)

    for jtype = 1:num_of_types
        itype = parse(Int64, split(readline(fp))[1])
        #println(itype)
        description = readline(fp)
        #println(description)
        atomtype = readline(fp)
        nenv = parse(Int64, split(readline(fp))[1])
        #println(nenv)
        envtypes = Vector{String}(undef, nenv)
        for k = 1:nenv
            u = split(readline(fp))[1]
            #println(u)
            envtypes[k] = u
        end
        #println(envtypes)
        rc_min = parse(Float64, split(readline(fp))[1])
        rc_max = parse(Float64, split(readline(fp))[1])
        #println((rc_min, rc_max))
        sftype = readline(fp)
        #println(sftype)
        nsf = parse(Int64, split(readline(fp))[1])
        nsfparam = parse(Int64, split(readline(fp))[1])
        #println((nsf, nsfparam))
        sf = parse.(Int64, split(readline(fp)))

        #println(sf)
        sfparam = zeros(Float64, nsfparam, nsf)
        sfparam[:] .= parse.(Float64, split(readline(fp)))
        #(sfparam)


        sfenv = zeros(Int64, 2, nsf)
        sfenv[:] .= parse.(Int64, split(readline(fp)))
        #display(sfenv)
        neval = parse(Int64, split(readline(fp))[1])

        sfval_min = zeros(Float64, nsf)
        sfval_min .= parse.(Float64, split(readline(fp)))
        sfval_max = zero(sfval_min)
        sfval_max .= parse.(Float64, split(readline(fp)))
        sfval_avg = zero(sfval_min)
        sfval_avg .= parse.(Float64, split(readline(fp)))
        sfval_cov = zero(sfval_min)
        sfval_cov .= parse.(Float64, split(readline(fp)))
        #display(sfval_min)
        #display(sfval_max)
        #display(sfval_avg)
        #display(sfval_cov)


        #println(readline(fp))

        fingerprints[itype] = FingerPrint(
            itype,
            description,
            atomtype,
            nenv,
            envtypes,
            rc_min,
            rc_max,
            sftype,
            nsf,
            nsfparam,
            sf,
            sfparam,
            sfenv,
            neval,
            sfval_min,
            sfval_max,
            sfval_avg,
            sfval_cov,
        )
        #error("dd")
    end

    headerposision = position(fp)
    #println("Position: ", pos)
    fileheader = filename
    datafilenames = Vector{String}(undef, num_of_structs)
    for istruc = 1:num_of_structs
        datafilenames[istruc] = fileheader * "_data" * lpad(istruc, 7, '0') * ".jld2"
    end

    E_max = min(E_max, maxenergy)

    E_scale = 2.0 / (E_max - E_min)
    E_shift = 0.5 * (E_max + E_min)

    fileused = zeros(Int64, num_of_structs)

    fingerprintstuple = NamedTuple{keys}(fingerprints)


    fingerprint_parameters_set =
        Vector{Vector{FingerPrintParams}}(undef, length(type_names))
    if data["numbasiskinds"] != 1
        for itype = 1:length(type_names)
            fingerprint_i = getfield(fingerprintstuple, keys[itype])
            fingerprint_parameters_set[itype] = get_multifingerprints_info(fingerprint_i)
        end
        println(fingerprint_parameters_set)
    else
        for itype = 1:length(type_names)
            fingerprint_i = getfield(fingerprintstuple, keys[itype])
            inputdim = fingerprint_i.nsf
            fingerprint_parameters_set[itype] =
                get_singlefingerprints_info(fingerprint_i, inputdim)
        end
    end


    numbasiskinds = data["numbasiskinds"]
    if numbasiskinds == 1
        num_of_structs2 = writefulldata_to_jld2(
            fp,
            headerposision,
            num_of_structs,
            joinpath(d, filename),
            type_names,
            E_shift,
            E_scale,
            datafilenames,
            fingerprintstuple,
            data["normalize"],
        )
        num_of_structs = num_of_structs2
    else
        num_of_structs2 = writefulldata_to_jld2_multi(
            data,
            fp,
            headerposision,
            num_of_structs,
            joinpath(d, filename),
            type_names,
            E_shift,
            E_scale,
            datafilenames,
            fingerprintstuple,
        )
        num_of_structs = num_of_structs2
    end

    dataffile = jldopen(joinpath(d, fileheader * ".jld2"), "r")

    dataset = BPDataset{
        keys,
        num_of_types,
        num_of_structs,
        typeof(fp),
        typeof(dataffile),
        numbasiskinds,
    }(
        filename,
        num_of_types, #::Int64
        num_of_structs, #::Int64
        type_names, #::Vector{String}
        E_atom, #::Vector{Float64}
        normalized, #::Bool
        E_scale, #::Float64
        E_shift, #::Float64
        natomtot, #::Int64
        E_avg, #::Float64
        E_min, #::Float64
        E_max, #::Float64
        has_setups, #::Bool
        fp, #::IOStream
        fingerprintstuple,
        #        NamedTuple{keys}(fingerprints),
        headerposision,
        datafilenames,
        fileused,
        dataffile,
        fingerprint_parameters_set,
    )

    println("------------------------------------------------")
    println("dataset: $filename ")
    println("num. of data: $num_of_structs")
    println("------------------------------------------------")

    return dataset, data

end

function get_coefficients(
    dataset::BPDataset{keys,num_of_types,num_of_structs},
    istruct::Integer,
    fp,
) where {keys,num_of_types,num_of_structs}
    #@assert istruct <= num_of_structs "size of the dataset $(num_of_structs) is smaller than the index $istruct"
    dataset.fileused[istruct] = true
    energy = 0.0
    coefficients = Vector{Matrix{Float64}}(undef, num_of_types)
    natoms = 0
    #jldopen(dataset.datafilenames[istruct], "r") do file
    energy = fp["$istruct"]["energy"]
    coefficients .= fp["$istruct"]["coefficients"]
    natoms = fp["$istruct"]["natoms"]
    #println((energy[1], coefficients, natoms[1]))
    #end
    #@load dataset.datafilenames[istruct] energy coefficients natoms
    #println(typeof(coefficients))
    #error("dd")
    return energy, coefficients, natoms
end

function get_unusedindex(dataset::BPDataset)
    return get_unusedindex_arr(dataset::BPDataset, dataset.fileused)
end

function get_unusedindex(dataset::BPDataset, dataindices)
    arr = view(dataset.fileused, dataindices)
    #@code_warntype get_unusedindex_arr(dataset::BPDataset, arr)
    #error("dg")
    return get_unusedindex_arr(dataset::BPDataset, arr)
end

function get_unusedindex_arr(dataset::BPDataset, arr)
    arr = dataset.fileused

    count_zeros = count(==(0), arr)

    if count_zeros == 0
        #@warn "no unused data. reload_data! should be perfomed."
        s = 0
        #return 0
    else
        zero_index = findnext(==(0), arr, 1)
        for i = 2:rand(1:count_zeros)
            zero_index = findnext(==(0), arr, zero_index + 1)
        end
        s = ifelse(zero_index == nothing, 0, zero_index)
        #zero_index
        #return zero_index
    end
    return s
end
