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
end

# ╔═╡ 799f9576-7cbd-4a0b-9136-807b0f55f5d2
using MLUtils: DataLoader

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

# ╔═╡ 27832226-9004-436a-9035-95ad00167c8e
numbatch = toml["numbatch"]

# ╔═╡ 357b6d0d-8dd4-4649-9ec8-4b0d3092e3a0
train_loader = DataLoader(traindata; batchsize=numbatch)

# ╔═╡ 32a68d5b-eb11-4873-aeda-2779d198f7a2
iterate(train_loader)

# ╔═╡ f0eaa849-88c0-436a-a791-fa09ee1d5b2c
toml["numbasiskinds"]

# ╔═╡ f7ff252f-88ec-4d2c-b2d3-00dba4c3d3f5
toml["atomtypes"]

# ╔═╡ f9642223-6f9d-48ea-85fe-a2791affe453
toml["kan"]

# ╔═╡ dc633991-f940-4ec8-8538-44ff2118904d
toml["resnet"]

# ╔═╡ Cell order:
# ╠═640037f4-43f8-11f0-1b23-910aaead95b2
# ╠═799f9576-7cbd-4a0b-9136-807b0f55f5d2
# ╠═b1745dc3-b967-4018-9054-32412e936d3c
# ╠═be060fa5-c382-4ef6-86a0-bc8ccefb7db1
# ╠═d038767e-c41f-46fa-8dc3-a8faf8ed85ec
# ╠═27832226-9004-436a-9035-95ad00167c8e
# ╠═357b6d0d-8dd4-4649-9ec8-4b0d3092e3a0
# ╠═32a68d5b-eb11-4873-aeda-2779d198f7a2
# ╠═f0eaa849-88c0-436a-a791-fa09ee1d5b2c
# ╠═f7ff252f-88ec-4d2c-b2d3-00dba4c3d3f5
# ╠═f9642223-6f9d-48ea-85fe-a2791affe453
# ╠═dc633991-f940-4ec8-8538-44ff2118904d
