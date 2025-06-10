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
	using Flux
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

# ╔═╡ 4a5acf2e-8d94-4ae5-b786-6acd9cd417e1
model = BPNetTrainer.FluxEdition.FluxBPNet(toml, bpdata.fingerprint_parameters) |> Flux.f64

# ╔═╡ 7c1ad009-6873-4670-8232-9574db0eb069
x, y = traindata[1]

# ╔═╡ 2adb7aeb-16ee-499d-a468-1a4c90ba5d8a
model(x)

# ╔═╡ 389ae298-8eb8-4531-9b24-76681deff0a3
begin
	numbatch = toml["numbatch"]
	train_loader = DataLoader(traindata; batchsize=numbatch)
	println("num. of training data $(length(traindata))")

	testdata = BPDataMemory(bpdata, filename_test)
	test_loader = DataLoader(testdata; batchsize=1)
	println("num. of testing data $(length(testdata))")
end

# ╔═╡ efc015ed-5617-4550-b5cc-d978f8e7a39d
begin
	θ, re = Flux.destructure(model)
	#grad = Flux.gradient(θ -> sum(re(θ)(x)), θ)
	#display(grad[1])

	state = BPNetTrainer.FluxEdition.set_state(toml, θ)
	lossfunction(x, y) = Flux.mse(x, y)
	# println(lossfunction(x, y))
	println("num. of parameters: $(length(θ))")


	nepoch = toml["nepoch"]
	@time BPNetTrainer.FluxEdition.training!(
		θ, re, state, train_loader, test_loader, lossfunction, nepoch; modelparamfile=toml["modelparamfile"]
	)
end

# ╔═╡ Cell order:
# ╠═640037f4-43f8-11f0-1b23-910aaead95b2
# ╠═799f9576-7cbd-4a0b-9136-807b0f55f5d2
# ╠═b1745dc3-b967-4018-9054-32412e936d3c
# ╠═be060fa5-c382-4ef6-86a0-bc8ccefb7db1
# ╠═d038767e-c41f-46fa-8dc3-a8faf8ed85ec
# ╠═4a5acf2e-8d94-4ae5-b786-6acd9cd417e1
# ╠═7c1ad009-6873-4670-8232-9574db0eb069
# ╠═2adb7aeb-16ee-499d-a468-1a4c90ba5d8a
# ╠═389ae298-8eb8-4531-9b24-76681deff0a3
# ╠═efc015ed-5617-4550-b5cc-d978f8e7a39d
