struct LayerAcceptsInputWithDataAndLabels{L} <: LuxCore.AbstractLuxContainerLayer{(:layer,)}
    layer::L
end

function (m::LayerAcceptsInputWithDataAndLabels)(x, ps, st::NamedTuple)
    y, st = Lux.apply(m.layer, Tuple(x.data), ps.layer, st.layer)
    y * x.labels, (; layer = st)
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
            d = Lux.Dense(h_in, h_out)
            push!(layers, d)
            continue
        else
            d = if h_in == h_out
                if resnet
                    Lux.Parallel(+, identity, Lux.Dense(h_in, h_out, activations[i]))
                else
                    Lux.Dense(h_in, h_out, activations[i])
                end
            else
                Lux.Dense(h_in, h_out, activations[i])
            end
            push!(layers, d)
        end
    end
    return Lux.Chain(layers...)
end

struct LuxBPNet{L} <: LuxCore.AbstractLuxContainerLayer{(:layer,)}
    layer::L
end

function LuxBPNet(toml::Dict{String,Any}, fingerprint_params)
    atomtypes = toml["atomtypes"]
    numbasiskinds = toml["numbasiskinds"]
    chains = map(enumerate(atomtypes)) do (itype, name)
        fingerprint_param = fingerprint_params[itype]

        itype_chains = Lux.Chain[]
        for ikind = 1:numbasiskinds
            inputdim = fingerprint_param[ikind].numparams
            hidden_dims = toml[name]["layers"]
            activations = getproperty.(Ref(Lux), Symbol.(toml[name]["activations"]))

            c = make_dense_chain(inputdim, hidden_dims, activations)
            push!(itype_chains, c)
        end
        # LayerAcceptsInputWithDataAndLabels(Parallel(+, itype_chains...))
        Parallel(*, Parallel(+, itype_chains...), NoOpLayer())
    end
    LuxBPNet(Parallel(+, chains...))
end

function (m::LuxBPNet)(x, ps, st::NamedTuple)
    y, st = Lux.apply(m.layer, Tuple(x),  ps.layer, st.layer)
    return y, (; layer=st)
end