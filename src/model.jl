struct FluxBPNet{Chains}
    chains::Chains
end

function apply_model(m, x)
    m(x)
end

function apply_bpmultimodel(ci, xi)
    ei = sum(map(apply_model, ci, xi.data))
    return ei * xi.labels
end

function (m::FluxBPNet)(x)
    energies = sum(eachindex(m.chains, axes(x, 1))) do i
        apply_bpmultimodel(m.chains[i], x[i])
    end
    return energies
end

function make_dense_chain(
    inputdim::Int,
    hidden_dims::Vector{Int},
    activations::AbstractVector,
    resnet = false,
)
    layers = []
    dims = [inputdim, hidden_dims..., 1]
    for i = 1:(length(dims)-1)
        h_in = dims[i]
        h_out = dims[i+1]
        if i == length(dims) - 1
            # last layer
            d = Flux.Dense(h_in, h_out)
            push!(layers, d)
            continue
        else
            d = if h_in == h_out
                if resnet
                    Flux.Parallel(+, identity, Dense(h_in, h_out, activations[i]))
                else
                    Flux.Dense(h_in, h_out, activations[i])
                end
            else
                Flux.Dense(h_in, h_out, activations[i])
            end
            push!(layers, d)
        end
    end
    return Flux.Chain(layers...)
end

function FluxBPNet(toml::Dict{String,Any})
    atomtypes = toml["atomtypes"]
    numbasiskinds = toml["numbasiskinds"]
    chains = map(enumerate(atomtypes)) do (itype, name)
        itype_chains = Flux.Chain[]
        for ikind = 1:numbasiskinds
            hidden_dims = toml[name]["layers"]
            activations = getproperty.(Ref(Flux), Symbol.(toml[name]["activations"]))

            inputdim = 32 # from fingerprint params
            c = make_dense_chain(inputdim, hidden_dims, activations)
            push!(itype_chains, c)
        end
        itype_chains
    end
    FluxBPNet(chains)
end
