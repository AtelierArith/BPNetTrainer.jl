

function full_gc_and_reclaim()
    GC.gc(true)
    MLDataDevices.functional(CUDADevice) && CUDA.reclaim()
    return nothing
end

"""
    ddptraining(distributed_backend)

Distributed data parallel training for a BPNet model using Lux.
"""
function ddptraining(distributed_backend, tomlpath)
    device = gpu_device()
    local_rank = Lux.DistributedUtils.local_rank(distributed_backend)
    total_workers = Lux.DistributedUtils.total_workers(distributed_backend)

    is_distributed = total_workers > 1
    should_log = !is_distributed || local_rank == 0

    sensible_println(msg) = should_log && println("[$(Dates.now())] ", msg)
    sensible_print(msg) = should_log && print("[$(Dates.now())] ", msg)


    if local_rank == 0
        bpdata, toml = BPDataset(tomlpath)
    else
        bpdata = nothing
        toml = nothing
    end

    bpdata = MPI.bcast(bpdata, 0, MPI.COMM_WORLD)
    toml = MPI.bcast(toml, 0, MPI.COMM_WORLD)

    ratio = toml["testratio"]
    filename_train = toml["filename_train"]
    filename_test = toml["filename_test"]
    if local_rank == 0
        traindata = BPDataMemory(bpdata, filename_train)
        testdata = BPDataMemory(bpdata, filename_test)
    else
        traindata = nothing
        testdata = nothing
        filename_train = nothing
        filename_test = nothing
    end
    traindata = MPI.bcast(traindata, 0, MPI.COMM_WORLD)
    testdata = MPI.bcast(testdata, 0, MPI.COMM_WORLD)

    if local_rank == 0
        make_train_and_test_jld2(bpdata, filename_train, filename_test; ratio)
    end

    train_E_scale = traindata.E_scale
    test_E_scale = testdata.E_scale

    numbatch = toml["numbatch"]

    traindata = Lux.DistributedUtils.DistributedDataContainer(
        distributed_backend,
        traindata,
    )

    batchsize = numbatch ÷ total_workers
    train_loader = MLUtils.DataLoader(
        traindata;
        batchsize,
        shuffle = true,
        partial = false,
        parallel = true,
    )

    testdata = Lux.DistributedUtils.DistributedDataContainer(
        distributed_backend,
        testdata,
    )

    test_loader = MLUtils.DataLoader(
        testdata;
        batchsize = 1,
        shuffle = false,
        partial = true,
        parallel = true,
    )

    model = LuxBPNet(toml, bpdata.fingerprint_parameters)

    rng = Xoshiro(1234)
    _ps, _st = Lux.setup(rng, model) |> device

    ps = DistributedUtils.synchronize!!(distributed_backend, _ps)
    st = DistributedUtils.synchronize!!(distributed_backend, _st)

    opt = DistributedUtils.DistributedOptimizer(
        distributed_backend,
        Optimisers.AdamW(),
    )
    full_gc_and_reclaim()

    tstate = Lux.Training.TrainState(model, ps, st, opt)
    lossfn = OnlyFollowsLossFn(Lux.MSELoss())
    nepoch = toml["nepoch"]
    for epoch = 1:nepoch
        should_log && @info epoch
        should_log && @info ("Training phase")
        st = Lux.trainmode(st)

        train_loss = 0.0
        train_sse = 0.0
        for (i, (x, y, num, totalnumatom)) in enumerate(train_loader)
            x_dev = [(Tuple(device(Lux.f32(e.data))), device(Lux.f32(e.labels))) for e in x]
            y_dev = y |> Lux.f32 |> device

            ŷ, _ = Lux.apply(model, x_dev, ps, st)

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
        train_rmse = sqrt(train_sse) / train_E_scale
        should_log && @info ("train loss: ", train_loss / length(train_loader))
        should_log && @info ("train rmse: ", train_rmse, "[eV/atom]")

        should_log && @info ("Validation phase")
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
        test_rmse = sqrt(test_sse) / test_E_scale
        should_log && @info ("test loss: ", test_loss / length(test_loader))
        should_log && @info ("test rmse: ", test_rmse / length(test_loader), "[eV/atom]")
    end
end
