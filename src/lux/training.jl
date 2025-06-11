
struct OnlyFollowsLossFn{F} <: Lux.AbstractLossFunction
    fn_internal::F
end

function (fn::OnlyFollowsLossFn)(ŷ, y)
    only(fn.fn_internal(ŷ, y))
end

function training(bpdata, toml)
    device = gpu_device()

    ratio = toml["testratio"]
    filename_train = toml["filename_train"]
    filename_test = toml["filename_test"]
    make_train_and_test_jld2(bpdata, filename_train, filename_test; ratio)

    numbatch = toml["numbatch"]
    traindata = BPDataMemory(bpdata, filename_train)
    train_loader = MLUtils.DataLoader(traindata; batchsize = numbatch, shuffle = true)

    testdata = BPDataMemory(bpdata, filename_test)
    test_loader = MLUtils.DataLoader(testdata; batchsize = 1)

    model = LuxBPNet(toml, bpdata.fingerprint_parameters)

    rng = Xoshiro(1234)
    _ps, _st = Lux.setup(rng, model)
    ps = _ps |> device
    st = _st |> device
    tstate = Lux.Training.TrainState(model, ps, st, Optimisers.AdamW())
    lossfn = OnlyFollowsLossFn(Lux.MSELoss())
    nepoch = toml["nepoch"]
    for epoch = 1:nepoch
        @info epoch
        @info ("Training phase")
        st = Lux.trainmode(st)

        train_loss = 0.0
        train_sse = 0.0
        for (i, (x, y, num, totalnumatom)) in enumerate(train_loader)
            x_dev = [(Tuple(device(Lux.f32(e.data))), device(Lux.f32(e.labels))) for e in x]
            y_dev = y |> Lux.f32 |> device

            # ŷ, _ = Lux.apply(model, x_dev, ps, st)
            _, loss, _, tstate = Lux.Training.single_train_step!(
                AutoZygote(),
                lossfn,
                (x_dev, y_dev),
                tstate,
            )
            train_loss += cpu_device()(loss)
            train_sse += loss / totalnumatom^2
        end
        train_sse = train_sse / length(train_loader)
        train_rmse = sqrt(train_sse) / train_loader.data.E_scale
        @info ("train loss: ", train_loss / length(train_loader))
        @info ("train rmse: ", train_rmse, "[eV/atom]")

        @info ("Validation phase")
        st = Lux.testmode(st)

        test_loss = 0.0
        test_sse = 0.0
        for (i, (x, y, num, totalnumatom)) in enumerate(test_loader)
            x_dev = [(Tuple(device(Lux.f32(e.data))), device(Lux.f32(e.labels))) for e in x]
            y_dev = y |> Lux.f32 |> device

            ŷ, _ = Lux.apply(model, x_dev, ps, st)
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
