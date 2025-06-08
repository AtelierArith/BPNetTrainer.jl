### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 640037f4-43f8-11f0-1b23-910aaead95b2
begin
    using Pkg
    using Revise
    Pkg.activate(dirname(dirname(@__DIR__)))
    using BPNetTrainer
    using BPNetTrainer: download_dataset, generate_example_dataset
    using BPNetTrainer: make_train_and_test_jld2
    using BPNetTrainer: adddata!, set_numfiles!, make_descriptor
    using BPNetTrainer: FingerPrint, DataGenerator, BPDataset, BPDataMemory
end

# ╔═╡ 799f9576-7cbd-4a0b-9136-807b0f55f5d2
begin
	using Random
	using Lux
	using MLUtils
end

# ╔═╡ b1745dc3-b967-4018-9054-32412e936d3c
begin
    download_dataset()
    generate_example_dataset()
    tomlpath = joinpath(pkgdir(BPNetTrainer), "configs", "test_input.toml")
    bpdata, toml = BPDataset(tomlpath)
end

# ╔═╡ be060fa5-c382-4ef6-86a0-bc8ccefb7db1
begin
    ratio = toml["testratio"]
    filename_train = toml["filename_train"]
    filename_test = toml["filename_test"]
    make_train_and_test_jld2(bpdata, filename_train, filename_test; ratio)
end

# ╔═╡ d038767e-c41f-46fa-8dc3-a8faf8ed85ec
traindata = BPDataMemory(bpdata, filename_train)

# ╔═╡ 69afd7b4-84de-4528-86ec-e479bc33e785
begin
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
	        LayerAcceptsInputWithDataAndLabels(Parallel(+, itype_chains...))
	    end
	    layers = Parallel(+, chains...)
	    LuxBPNet(layers)
	end
	
	function (m::LuxBPNet)(x, ps, st::NamedTuple)
	    y, st = Lux.apply(m.layer, Tuple(x),  ps.layer, st.layer)
	    return y, (; layer=st)
	end
end

# ╔═╡ 5c15f7d1-0c09-49c7-80a1-3c0e6834a84f
model = LuxBPNet(toml, bpdata.fingerprint_parameters)

# ╔═╡ 7c1ad009-6873-4670-8232-9574db0eb069
x, y = traindata[1]

# ╔═╡ 0a2ea050-b4e6-499c-aaeb-b4cded9e3fca
x

# ╔═╡ ef934b83-c651-4b97-ae6f-f8def737f28b
let
	rng = Xoshiro(1234)
	ps, st = Lux.setup(rng, model)
	Lux.apply(model, [(data = Lux.f32(e.data), labels=Lux.f32(e.labels)) for e in x], ps, st)
end

# ╔═╡ Cell order:
# ╠═640037f4-43f8-11f0-1b23-910aaead95b2
# ╠═799f9576-7cbd-4a0b-9136-807b0f55f5d2
# ╠═b1745dc3-b967-4018-9054-32412e936d3c
# ╠═be060fa5-c382-4ef6-86a0-bc8ccefb7db1
# ╠═d038767e-c41f-46fa-8dc3-a8faf8ed85ec
# ╠═69afd7b4-84de-4528-86ec-e479bc33e785
# ╠═5c15f7d1-0c09-49c7-80a1-3c0e6834a84f
# ╠═7c1ad009-6873-4670-8232-9574db0eb069
# ╠═0a2ea050-b4e6-499c-aaeb-b4cded9e3fca
# ╠═ef934b83-c651-4b97-ae6f-f8def737f28b
