### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 640037f4-43f8-11f0-1b23-910aaead95b2
begin
    using Pkg
    using Revise
    Pkg.activate(dirname(dirname(@__DIR__)))
    using LuxBPNet
    using LuxBPNet: download_dataset, generate_example_dataset
    using LuxBPNet: make_train_and_test_jld2
    using LuxBPNet: adddata!, set_numfiles!, make_descriptor
    using LuxBPNet: FingerPrint, DataGenerator, BPDataset, BPDataMemory
	using LuxBPNet: FluxBPNet
end

# ╔═╡ 799f9576-7cbd-4a0b-9136-807b0f55f5d2
using Flux

# ╔═╡ b1745dc3-b967-4018-9054-32412e936d3c
begin
    download_dataset()
    generate_example_dataset()
    tomlpath = joinpath(pkgdir(LuxBPNet), "configs", "test_input.toml")
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
model = FluxBPNet(toml) |> Flux.f64

# ╔═╡ 7c1ad009-6873-4670-8232-9574db0eb069
x, y = traindata[1]

# ╔═╡ 2adb7aeb-16ee-499d-a468-1a4c90ba5d8a
model(x)

# ╔═╡ Cell order:
# ╠═640037f4-43f8-11f0-1b23-910aaead95b2
# ╠═799f9576-7cbd-4a0b-9136-807b0f55f5d2
# ╠═b1745dc3-b967-4018-9054-32412e936d3c
# ╠═be060fa5-c382-4ef6-86a0-bc8ccefb7db1
# ╠═d038767e-c41f-46fa-8dc3-a8faf8ed85ec
# ╠═4a5acf2e-8d94-4ae5-b786-6acd9cd417e1
# ╠═7c1ad009-6873-4670-8232-9574db0eb069
# ╠═2adb7aeb-16ee-499d-a468-1a4c90ba5d8a
