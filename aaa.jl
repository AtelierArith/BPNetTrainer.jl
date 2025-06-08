### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 640037f4-43f8-11f0-1b23-910aaead95b2
begin
	using Pkg
	using Revise
	Pkg.activate(@__DIR__)
	using LuxBPNet
	using LuxBPNet: download_dataset, generate_example_dataset
	using LuxBPNet: make_train_and_test_jld2
	using LuxBPNet: adddata!, set_numfiles!, make_descriptor
	using LuxBPNet: FingerPrint, DataGenerator, BPDataset
end

# ╔═╡ b1745dc3-b967-4018-9054-32412e936d3c
begin
	download_dataset()
	generate_example_dataset()
	tomlpath = joinpath("configs", "test_input.toml")
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
filename_test

# ╔═╡ 27832226-9004-436a-9035-95ad00167c8e
filename_train

# ╔═╡ Cell order:
# ╠═640037f4-43f8-11f0-1b23-910aaead95b2
# ╠═b1745dc3-b967-4018-9054-32412e936d3c
# ╠═be060fa5-c382-4ef6-86a0-bc8ccefb7db1
# ╠═d038767e-c41f-46fa-8dc3-a8faf8ed85ec
# ╠═27832226-9004-436a-9035-95ad00167c8e
