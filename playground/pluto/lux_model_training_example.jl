### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 640037f4-43f8-11f0-1b23-910aaead95b2
begin
    using Pkg
    using Revise
    Pkg.activate(dirname(dirname(@__DIR__)))
    Pkg.instantiate()

    using Random
    using Statistics
    using Lux
    using MLUtils
    using Optimisers
    using LuxCUDA

    using BPNetTrainer
    using BPNetTrainer: download_dataset, generate_example_dataset
    using BPNetTrainer: make_train_and_test_jld2
    using BPNetTrainer: adddata!, set_numfiles!, make_descriptor
    using BPNetTrainer: FingerPrint, DataGenerator, BPDataset, BPDataMemory
end

# ╔═╡ b1745dc3-b967-4018-9054-32412e936d3c
begin
    download_dataset()
    generate_example_dataset()
    tomlpath = joinpath(pkgdir(BPNetTrainer), "configs", "test_input.toml")
    bpdata, toml = BPDataset(tomlpath)
end

# ╔═╡ 381c35ae-3132-4dfb-8ae8-41061eb08deb
@time BPNetTrainer.LuxEdition.training(bpdata, toml)

# ╔═╡ 6c24b87a-d6fd-4be6-8bcb-3f1ebd7d0213


# ╔═╡ Cell order:
# ╠═640037f4-43f8-11f0-1b23-910aaead95b2
# ╠═b1745dc3-b967-4018-9054-32412e936d3c
# ╠═381c35ae-3132-4dfb-8ae8-41061eb08deb
# ╠═6c24b87a-d6fd-4be6-8bcb-3f1ebd7d0213
