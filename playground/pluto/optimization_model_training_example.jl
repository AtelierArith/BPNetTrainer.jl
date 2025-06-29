### A Pluto.jl notebook ###
# v0.20.10

using Markdown
using InteractiveUtils

# ╔═╡ 640037f4-43f8-11f0-1b23-910aaead95b2
begin
    using Pkg
    using Revise
    Pkg.activate(dirname(dirname(@__DIR__)))
    Pkg.instantiate()

    using Random
    using Lux
    using MLUtils
    using Optimisers
	using ComponentArrays
	using OptimizationOptimJL
	import Zygote
	import Optim
	
    using BPNetTrainer
    using BPNetTrainer: download_dataset, generate_example_dataset, make_train_and_test_jld2
    using BPNetTrainer: BPDataset, BPDataMemory 
	using BPNetTrainer.LuxEdition: LuxBPNet
end

# ╔═╡ b1745dc3-b967-4018-9054-32412e936d3c
begin
    download_dataset()
    generate_example_dataset()
    tomlpath = joinpath(pkgdir(BPNetTrainer), "configs", "test_input.toml")
    bpdata, toml = BPDataset(tomlpath)
end

# ╔═╡ 4bf48f42-b084-455a-b868-94d1d1351901
begin
	struct OnlyFollowsLossFn{F} <: Lux.AbstractLossFunction
	    fn_internal::F
	end
	
	function (fn::OnlyFollowsLossFn)(ŷ, y)
	    only(fn.fn_internal(ŷ, y))
	end
		
	lossfn = OnlyFollowsLossFn(Lux.MSELoss())
end

# ╔═╡ 381c35ae-3132-4dfb-8ae8-41061eb08deb
begin
	device = gpu_device()
	
	ratio = toml["testratio"]
	filename_train = toml["filename_train"]
	filename_test = toml["filename_test"]
	make_train_and_test_jld2(bpdata, filename_train, filename_test; ratio)
	
	numbatch = toml["numbatch"]
end

# ╔═╡ bf703ec9-6e56-433e-ba70-ac92f8ad62ad
begin
	traindata = BPDataMemory(bpdata, filename_train)
	train_loader = MLUtils.DataLoader(
		traindata; batchsize = length(traindata), shuffle = true
	)
	
	testdata = BPDataMemory(bpdata, filename_test)
	test_loader = MLUtils.DataLoader(testdata; batchsize = 1)
	
	model = LuxBPNet(toml, bpdata.fingerprint_parameters)
	
	rng = Xoshiro(1234)
	_ps, _st = Lux.setup(rng, model)
	ps = _ps |> device
	st = _st |> device
	ps_ca = ComponentArray(ps)
	stateful_model = StatefulLuxLayer{true}(model, nothing, st)
end

# ╔═╡ b2ae53f9-d6f8-42fd-a121-cae049fc7c1a
begin
	function _loss_function(p, batch)
		(x, y, num, totalnumatom) = batch 
	    x_dev = [
			(Tuple(device(Lux.f32(e.data))), device(Lux.f32(e.labels))) for e in x
		]
        y_dev = y |> Lux.f32 |> device
        ŷ = Lux.apply(stateful_model, x_dev, p)
        loss = lossfn(ŷ, y_dev)
        return cpu_device()(loss)
	end
	
	function loss_function(p, dataloader::MLUtils.DataLoader)
        loss = 0
		sse = 0
		for batch in dataloader
			(x, y, num, totalnumatom) = batch 
			loss += _loss_function(p, batch)	
			sse += loss / totalnumatom^2
		end
		return loss
	end
end

# ╔═╡ e2dc5acb-1a83-4f23-aeca-4025dc794743
optfun = OptimizationFunction(
    loss_function,
    Optimization.AutoZygote()
)

# ╔═╡ 539b3f61-3f58-4f17-85e1-30043a161b35
let
	function callback(state, l) #callback function to observe training
	    # display(l)
		@info l
	    return false
	end
	
	optprob = OptimizationProblem(optfun, ps_ca, train_loader)
	for e in 1:5
		@info "epoch: $(e)"

		res_with_optim = Optimization.solve(
		    optprob, Optim.LBFGS(), 
			callback = callback,
			allow_f_increases = false,
		)
		optprob = remake(optprob; u0 = res_with_optim.u)
		# Evaluate Training dataset
		let
		    train_loss = 0.0
	        train_sse = 0.0
	        for (i, (x, y, num, totalnumatom)) in enumerate(train_loader)
	            x_dev = [(Tuple(device(Lux.f32(e.data))), device(Lux.f32(e.labels))) for e in x]
	            y_dev = y |> Lux.f32 |> device
	
	            ŷ = Lux.apply(stateful_model, x_dev, res_with_optim.u)
	            loss = lossfn(ŷ, y_dev)
	            train_loss += cpu_device()(loss)
	            train_sse += loss / totalnumatom^2
	        end
	        train_sse = train_sse / length(train_loader)
	        train_rmse = sqrt(train_sse) / train_loader.data.E_scale
			@info ("test loss: ", train_loss / length(train_loader))
	        @info ("test rmse: ", train_rmse / length(train_loader), "[eV/atom]")
		end
		
		# Evaluate Test dataset
		let
		    test_loss = 0.0
	        test_sse = 0.0
	        for (i, (x, y, num, totalnumatom)) in enumerate(test_loader)
	            x_dev = [(Tuple(device(Lux.f32(e.data))), device(Lux.f32(e.labels))) for e in x]
	            y_dev = y |> Lux.f32 |> device
	
	            ŷ = Lux.apply(stateful_model, x_dev, res_with_optim.u)
	            loss = lossfn(ŷ, y_dev)
	            test_loss += cpu_device()(loss)
	            test_sse += loss / totalnumatom^2
	        end
	        test_sse = test_sse / length(test_loader)
	        test_rmse = sqrt(test_sse) / test_loader.data.E_scale
			@info ("test loss: ", test_loss / length(test_loader))
	        @info ("test rmse: ", test_rmse / length(test_loader), "[eV/atom]")
		end
	end
end

# ╔═╡ Cell order:
# ╠═640037f4-43f8-11f0-1b23-910aaead95b2
# ╠═b1745dc3-b967-4018-9054-32412e936d3c
# ╠═4bf48f42-b084-455a-b868-94d1d1351901
# ╠═381c35ae-3132-4dfb-8ae8-41061eb08deb
# ╠═bf703ec9-6e56-433e-ba70-ac92f8ad62ad
# ╠═b2ae53f9-d6f8-42fd-a121-cae049fc7c1a
# ╠═e2dc5acb-1a83-4f23-aeca-4025dc794743
# ╠═539b3f61-3f58-4f17-85e1-30043a161b35
