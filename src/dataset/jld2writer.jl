function write_data_to_jld2(dataset::BPDataset, I::AbstractVector, filename)
    fp = dataset.datafile
    f = jldopen(filename, "w")
    num = length(I)
    for i = 1:num
        istruc = I[i]

        mygroup = JLD2.Group(f, "$i")
        mygroup["datafilename"] = fp["$istruc"]["datafilename"]
        mygroup["filename"] = fp["$istruc"]["filename"]
        mygroup["natoms"] = fp["$istruc"]["natoms"]
        mygroup["ntypes"] = fp["$istruc"]["ntypes"]
        mygroup["energy"] = fp["$istruc"]["energy"]
        mygroup["coefficients"] = fp["$istruc"]["coefficients"]
        mygroup["coordinates"] = fp["$istruc"]["coordinates"]
        mygroup["forces"] = fp["$istruc"]["forces"]
        mygroup["atomtypes"] = fp["$istruc"]["atomtypes"]
    end
    f["num_of_structs"] = num
    close(f)
end

#function writefulldata_to_jld2(dataset::BPDataset)
function writefulldata_to_jld2(fp, headerposision, num_of_structs, fileheader,
    type_names, E_shift, E_scale, datafilenames, fingerprints, normalized)
    #seek(dataset.fp, dataset.headerposision)
    seek(fp, headerposision)
    #num_of_structs = dataset.num_of_structs
    #fp = dataset.fp
    #fileheader = dataset.filename

    f = jldopen(fileheader * ".jld2", "w")

    #keys = Tuple(Symbol.(dataset.type_names))
    keys = Tuple(Symbol.(type_names))
    istruc2 = 0

    println("The network output energy will be normalized to the interval [-1,1].")
    println("  Energy scaling factor: f = ", E_scale)
    println("  Atomic energy shift  : s = ", E_shift)

    for istruc = 1:num_of_structs
        filelen = parse(Int64, readline(fp))
        #println(filelen)
        filename = readline(fp)
        #println(filename)
        natoms, ntypes = parse.(Int64, split(readline(fp)))
        #println((natoms, ntypes))
        energy = parse(Float64, readline(fp))
        #energy = dataset.E_scale * (energy - natoms * dataset.E_shift)
        energy = E_scale * (energy - natoms * E_shift)
        #println(energy / natoms)


        #println(energy)
        coefficients_atoms = Vector{Vector{Float64}}(undef, natoms)
        coordinates_atoms = zeros(Float64, 3, natoms)
        forces_atoms = zeros(Float64, 3, natoms)
        atomtypes = zeros(Int64, natoms)

        #datafilename = dataset.datafilenames[istruc]
        datafilename = datafilenames[istruc]
        index_itype = zeros(Int64, ntypes)
        nsf_itype = zeros(Int64, ntypes)

        for iatom = 1:natoms
            itype = parse(Int64, readline(fp))
            index_itype[itype] += 1
            atomtypes[iatom] = itype
            coordinates_atoms[:, iatom] = parse.(Float64, split(readline(fp)))
            forces_atoms[:, iatom] = parse.(Float64, split(readline(fp)))
            nsf = parse(Int64, readline(fp))
            nsf_itype[itype] = nsf
            coefficients_atoms[iatom] = parse.(Float64, split(readline(fp)))
            #println((itype, cooCart, forCart, nsf, sfval))
        end

        coefficients = Vector{Matrix{Float64}}(undef, ntypes)
        coordinates = Vector{Matrix{Float64}}(undef, ntypes)
        forces = Vector{Matrix{Float64}}(undef, ntypes)
        for itype = 1:ntypes
            coefficients[itype] = zeros(Float64, nsf_itype[itype], index_itype[itype])
            coordinates[itype] = zeros(Float64, 3, index_itype[itype])
            forces[itype] = zeros(Float64, 3, index_itype[itype])
        end

        index_itype .= 0
        for iatom = 1:natoms
            itype = atomtypes[iatom]
            #fingerprint = getfield(dataset.fingerprints, keys[itype])
            fingerprint = getfield(fingerprints, keys[itype])

            index_itype[itype] += 1
            coordinates[itype][:, index_itype[itype]] = coordinates_atoms[:, iatom]
            forces[itype][:, index_itype[itype]] = forces_atoms[:, iatom]

            if normalized
                shifts = fingerprint.sfval_avg
                #shifts = dataset.fingerprints[itype].sfval_avg
                s = sqrt.(fingerprint.sfval_cov .- shifts .* shifts)
                #println("$iatom,$s")
                scales = zero(s)
                for k = 1:length(s)
                    if s[k] != 0.0
                        scales[k] = 1 / s[k]
                    else
                        scales[k] = 1
                    end
                end
                #scales = 1 ./ s

                #index_itype[itype] += 1
                coefficients[itype][:, index_itype[itype]] = scales .* (coefficients_atoms[iatom] .- shifts)
            else
                coefficients[itype][:, index_itype[itype]] = coefficients_atoms[iatom][:]
            end
            #coordinates[itype][:, index_itype[itype]] = coordinates_atoms[:, iatom]
            #forces[itype][:, index_itype[itype]] = forces_atoms[:, iatom]
        end

        if energy / natoms > 1.001
            continue
        else
            istruc2 += 1

            mygroup = JLD2.Group(f, "$istruc2")
            mygroup["datafilename"] = datafilename
            mygroup["filename"] = filename
            mygroup["natoms"] = natoms
            mygroup["ntypes"] = ntypes
            mygroup["energy"] = energy
            mygroup["coefficients"] = coefficients
            mygroup["coordinates"] = coordinates
            mygroup["forces"] = forces
            mygroup["atomtypes"] = atomtypes
        end

        #@save datafilename filename natoms ntypes energy coefficients coordinates forces atomtypes

    end
    if istruc2 < num_of_structs
        println("$(num_of_structs- istruc2) high-energy structures are skipped")
    end
    f["num_of_structs"] = istruc2#num_of_structs
    close(f)

    return istruc2
end


function writefulldata_to_jld2_multi(data, fp, headerposision, num_of_structs, fileheader,
    type_names, E_shift, E_scale, datafilenames, fingerprints)
    #seek(dataset.fp, dataset.headerposision)
    seek(fp, headerposision)
    #num_of_structs = dataset.num_of_structs
    #fp = dataset.fp
    #fileheader = dataset.filename

    f = jldopen(fileheader * ".jld2", "w")

    #keys = Tuple(Symbol.(dataset.type_names))
    keys = Tuple(Symbol.(type_names))
    istruc2 = 0

    println("The network output energy will be normalized to the interval [-1,1].")
    println("  Energy scaling factor: f = ", E_scale)
    println("  Atomic energy shift  : s = ", E_shift)
    fingerprint_parameters_set = Vector{Vector{FingerPrintParams}}(undef, length(type_names))

    for itype = 1:length(type_names)
        fingerprint = getfield(fingerprints, keys[itype])
        fingerprint_parameters_set[itype] = get_multifingerprints_info(fingerprint)

    end
    println(fingerprint_parameters_set)
    #error("ddd")

    for istruc = 1:num_of_structs
        filelen = parse(Int64, readline(fp))
        #println(filelen)
        filename = readline(fp)
        #println(filename)
        natoms, ntypes = parse.(Int64, split(readline(fp)))
        #println((natoms, ntypes))
        energy = parse(Float64, readline(fp))
        #energy = dataset.E_scale * (energy - natoms * dataset.E_shift)
        energy = E_scale * (energy - natoms * E_shift)
        #println(energy / natoms)


        #println(energy)
        coefficients_atoms = Vector{Vector{Float64}}(undef, natoms)
        coordinates_atoms = zeros(Float64, 3, natoms)
        forces_atoms = zeros(Float64, 3, natoms)
        atomtypes = zeros(Int64, natoms)

        #datafilename = dataset.datafilenames[istruc]
        datafilename = datafilenames[istruc]
        index_itype = zeros(Int64, ntypes)
        nsf_itype = zeros(Int64, ntypes)

        for iatom = 1:natoms
            itype = parse(Int64, readline(fp))
            #println(data["atomtypes"][itype])
            #println(getfield(fingerprints, Symbol(data["atomtypes"][itype])).sfparam[:, 1])
            #error("dd")
            index_itype[itype] += 1
            atomtypes[iatom] = itype
            coordinates_atoms[:, iatom] = parse.(Float64, split(readline(fp)))
            forces_atoms[:, iatom] = parse.(Float64, split(readline(fp)))
            nsf = parse(Int64, readline(fp))
            nsf_itype[itype] = nsf
            coefficients_atoms[iatom] = parse.(Float64, split(readline(fp)))
            #println((itype, cooCart, forCart, nsf, sfval))
        end

        coefficients = Vector{Vector{Matrix{Float64}}}(undef, ntypes)

        coordinates = Vector{Matrix{Float64}}(undef, ntypes)
        forces = Vector{Matrix{Float64}}(undef, ntypes)
        for itype = 1:ntypes
            fparams = fingerprint_parameters_set[itype]
            #display(data[string(keys[itype])])
            num_kinds = length(fparams)
            coefficients[itype] = Vector{Matrix{Float64}}(undef, num_kinds)
            for ikind = 1:num_kinds
                numparams = fparams[ikind].numparams
                coefficients[itype][ikind] = zeros(Float64, numparams, index_itype[itype])
            end


            coordinates[itype] = zeros(Float64, 3, index_itype[itype])
            forces[itype] = zeros(Float64, 3, index_itype[itype])
        end

        index_itype .= 0
        for iatom = 1:natoms
            itype = atomtypes[iatom]
            #fingerprint = getfield(dataset.fingerprints, keys[itype])
            fingerprint = getfield(fingerprints, keys[itype])
            fparams = fingerprint_parameters_set[itype]
            #display(data[string(keys[itype])])
            data_itype = data[string(keys[itype])]

            num_kinds = length(fparams)

            index_itype[itype] += 1
            coordinates[itype][:, index_itype[itype]] = coordinates_atoms[:, iatom]
            forces[itype][:, index_itype[itype]] = forces_atoms[:, iatom]

            for ikind = 1:num_kinds
                fparams_ikind = fparams[ikind]
                startindex = fparams_ikind.startindex
                endindex = fparams_ikind.endindex

                #println(data_itype["normalize"][ikind])
                if data_itype["normalize"][ikind] == true
                    shifts = fingerprint.sfval_avg[startindex:endindex]
                    #shifts = dataset.fingerprints[itype].sfval_avg
                    s = sqrt.(fingerprint.sfval_cov[startindex:endindex] .- shifts .* shifts)
                    #println("$iatom,$s")
                    scales = zero(s)
                    for k = 1:length(s)
                        if s[k] != 0.0
                            scales[k] = 1 / s[k]
                        else
                            scales[k] = 1
                        end
                    end
                    coefficients[itype][ikind][:, index_itype[itype]] = scales .* (coefficients_atoms[iatom][startindex:endindex] .- shifts)
                else
                    coefficients[itype][ikind][:, index_itype[itype]] = coefficients_atoms[iatom][startindex:endindex]
                end
                #display(coefficients[itype][ikind][:, index_itype[itype]])
            end

            #error("dd")

            #scales = 1 ./ s

            #index_itype[itype] += 1
            #coefficients[itype][:, index_itype[itype]] = scales .* (coefficients_atoms[iatom] .- shifts)

        end

        if energy / natoms > 1.001
            continue
        else
            istruc2 += 1

            mygroup = JLD2.Group(f, "$istruc2")
            mygroup["datafilename"] = datafilename
            mygroup["filename"] = filename
            mygroup["natoms"] = natoms
            mygroup["ntypes"] = ntypes
            mygroup["energy"] = energy
            mygroup["coefficients"] = coefficients
            mygroup["coordinates"] = coordinates
            mygroup["forces"] = forces
            mygroup["atomtypes"] = atomtypes
        end

        #@save datafilename filename natoms ntypes energy coefficients coordinates forces atomtypes

    end
    if istruc2 < num_of_structs
        println("$(num_of_structs- istruc2) high-energy structures are skipped")
    end
    f["num_of_structs"] = istruc2#num_of_structs
    close(f)
    #error("dddd")

    return istruc2
end

function make_train_and_test_jld2(dataset::BPDataset, filename_train, filename_test; ratio=0.1)
    numsize = length(dataset)
    numtest = Int(ceil(numsize * ratio))
    numtrain = numsize - numtest
    randomindices = shuffle(1:numsize)
    trainindices = randomindices[1:numtrain]
    testindices = randomindices[numtrain+1:end]

    write_data_to_jld2(dataset, trainindices, joinpath(DATASET_ROOT[], filename_train))
    write_data_to_jld2(dataset, testindices, joinpath(DATASET_ROOT[], filename_test))
end
