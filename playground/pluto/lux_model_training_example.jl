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
	using Statistics
	using Lux
	using MLUtils
	using Optimisers
	using Enzyme
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

# ╔═╡ ef934b83-c651-4b97-ae6f-f8def737f28b
let
	traindata = BPDataMemory(bpdata, filename_train)
	model = BPNetTrainer.LuxEdition.LuxBPNet(
		toml, bpdata.fingerprint_parameters
	)
	x, y = traindata[1]
	rng = Xoshiro(1234)
	ps, st = Lux.setup(rng, model)
	Lux.apply(model, [(data = Lux.f32(e.data), labels=Lux.f32(e.labels)) for e in x], ps, st)
end

# ╔═╡ 823081d9-aa12-4f58-a953-2bd2d5bba28d
begin
	struct OnlyFollowsLossFn{F} <: Lux.AbstractLossFunction 
		fn_internal::F
	end
	
	function (fn::OnlyFollowsLossFn)(ŷ, y)
	    only(fn.fn_internal(ŷ, y))
	end
end

# ╔═╡ 012a5350-78ff-4717-8884-b5b5319fde2b
begin
	function luxtraining()
		device = gpu_device()
		numbatch = toml["numbatch"]
		traindata = BPDataMemory(bpdata, filename_train)
		train_loader = MLUtils.DataLoader(
			traindata; batchsize=numbatch, shuffle=true
		)
	
		testdata = BPDataMemory(bpdata, filename_test)
		test_loader = MLUtils.DataLoader(
			testdata; batchsize=1
		)
	
		model = BPNetTrainer.LuxEdition.LuxBPNet(
			toml, bpdata.fingerprint_parameters
		)
	
		rng = Xoshiro(1234)
		_ps, _st = Lux.setup(rng, model)
		ps = _ps |> device
		st = _st |> device
		tstate = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))
		lossfn = OnlyFollowsLossFn(Lux.MSELoss())
		nepoch = toml["nepoch"]
	
		for e in 1:100
			st = Lux.trainmode(st)
	
			train_losses = []
			for (i, (x, y)) in enumerate(train_loader)
				x_dev = [
					(; data = device(Lux.f32(e.data)), 
					   labels = device(Lux.f32(e.labels))
					) for e in x
				]
				y_dev = y |> Lux.f32 |> device
	
				ŷ, _ = Lux.apply(model, x_dev, ps, st)
				_, loss, _, tstate = Lux.Training.single_train_step!(
	                AutoEnzyme(), lossfn, (x_dev, y_dev), tstate
	            )
				push!(train_losses, cpu_device()(loss))
			end
			println("train loss: ", sum(train_losses) / length(train_loader))
	
			st = Lux.testmode(st)
			test_losses = map(enumerate(test_loader)) do (i, (x, y))
				x_dev = [
					(; data = device(Lux.f32(e.data)), 
					   labels = device(Lux.f32(e.labels))
					) for e in x
				]
				y_dev = y |> Lux.f32 |> device
	
				ŷ, _ = Lux.apply(model, x_dev, ps, st)
				loss = lossfn(ŷ, y) |> cpu_device()
			end
				
			println("test loss: ", sum(test_losses) / length(test_loader))
		end
	end
	
	@time luxtraining()
end

# ╔═╡ Cell order:
# ╠═640037f4-43f8-11f0-1b23-910aaead95b2
# ╠═799f9576-7cbd-4a0b-9136-807b0f55f5d2
# ╠═b1745dc3-b967-4018-9054-32412e936d3c
# ╠═be060fa5-c382-4ef6-86a0-bc8ccefb7db1
# ╠═ef934b83-c651-4b97-ae6f-f8def737f28b
# ╠═823081d9-aa12-4f58-a953-2bd2d5bba28d
# ╠═012a5350-78ff-4717-8884-b5b5319fde2b
