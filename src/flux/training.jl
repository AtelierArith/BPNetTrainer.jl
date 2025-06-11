#using cuDNN
#using CUDA
using Flux


using Optimisers

function set_state(inputdata, θ)
    if inputdata["optimiser"] == "AdamW"
        state = Optimisers.setup(Optimisers.AdamW(), θ)
    elseif inputdata["optimiser"] == "Adam"
        state = Optimisers.setup(Optimisers.Adam(), θ)
    else
        error("$(inputdata["optimiser"]) is not suported")
    end
    return state
end
export set_state


function trainprocess!(θ, re, state, train_loader, lossfunction)
    loss = 0.0
    for (x, y, num, totalnumatom) in train_loader
        #xd = (data = fmap(CuArray{Float64},x.data), labels= fmap(CuArray{Bool},x.labels))
        xd = fmap(CuArray{Float64}, x)
        yd = fmap(CuArray{Float64}, y)
        L, dLdθ = Flux.withgradient(θ -> lossfunction(re(θ)(xd), yd), θ)
        Optimisers.update!(state, θ, dLdθ[1])
        loss += L
    end
    loss = loss / length(train_loader)
    return loss
end

function testprocess(θ, re, state, test_loader, lossfunction)
    loss = 0.0
    rmse = 0.0
    sse = 0.0
    for (x, y, num, totalnumatom) in test_loader
        xd = fmap(CuArray{Float64}, x)
        yd = fmap(CuArray{Float64}, y)
        y_pred = re(θ)(xd) |> Flux.cpu_device()
        loss_t = lossfunction(y_pred, y)
        loss += loss_t
        rmse += sqrt(loss_t) / (totalnumatom * test_loader.data.E_scale)
        sse += loss_t / totalnumatom^2
    end
    loss = loss / length(test_loader)
    sse = sse / length(test_loader)
    return loss, rmse, sse
end

function trainprocess!(θ::Vector{Float64}, re, state, train_loader, lossfunction)
    loss = 0.0
    for (x, y, num, totalnumatom) in train_loader
        #println(typeof(x))
        #xd = (data = fmap(CuArray{Float64},x.data), labels= fmap(CuArray{Bool},x.labels))
        L, dLdθ = Flux.withgradient(θ -> lossfunction(re(θ)(x), y), θ)
        Optimisers.update!(state, θ, dLdθ[1])
        loss += L / (totalnumatom * train_loader.data.E_scale)^2
    end
    loss = loss / length(train_loader)
    return loss
end

function testprocess(θ::Vector{Float64}, re, state, test_loader, lossfunction)
    loss = 0.0
    rmse = 0.0
    sse = 0.0
    for (x, y, num, totalnumatom) in test_loader
        #display(x)
        #println(typeof(x))
        y_pred = re(θ)(x)
        loss_t = lossfunction(y_pred, y)
        loss += loss_t
        rmse += sqrt(loss_t) / (totalnumatom * test_loader.data.E_scale)
        sse += loss_t / totalnumatom^2# * test_loader.data.E_scale)
    end
    loss = loss / length(test_loader)
    sse = sse / length(test_loader)
    return loss, rmse, sse
end

"""
    training!(θ, re, state, train_loader, test_loader, lossfunction, nepoch; modelparamfile="tempmodelparams_flux.jld2")

Execute the main training loop for Flux-based neural networks.

This function runs the complete training process including forward passes, 
backpropagation, parameter updates, and periodic evaluation on test data.
It saves model parameters after each epoch for recovery and analysis.

# Arguments
- `θ`: Model parameters (typically from `Flux.params(model)`)
- `re`: Model reconstruction function 
- `state`: Optimizer state (from `set_state`)
- `train_loader`: DataLoader for training data
- `test_loader`: DataLoader for test/validation data  
- `lossfunction`: Loss function to minimize
- `nepoch::Int`: Number of training epochs

# Keyword Arguments
- `modelparamfile::String="tempmodelparams_flux.jld2"`: File to save model parameters

# Process
For each epoch:
1. Executes training step using `trainprocess!`
2. Evaluates on test data using `testprocess`
3. Prints progress (epoch, train loss, test RMSE, test SSE)
4. Saves current model parameters to JLD2 file

# Output Format
```
Epoch: 1, Train Loss: 0.1234, Test RMSE: 0.0567, Test SSE: 0.0890
```

# Example Usage
```julia
# Setup (typically done in higher-level training function)
θ = Flux.params(model)
state = set_state(inputdata, θ)
lossfunction = Flux.mse

# Run training
training!(θ, model, state, train_loader, test_loader, lossfunction, 100)
```

# See also
- [`trainprocess!`](@ref): Single training epoch implementation
- [`testprocess`](@ref): Test evaluation implementation  
- [`set_state`](@ref): Optimizer state initialization
"""
function training!(
    θ,
    re,
    state,
    train_loader,
    test_loader,
    lossfunction,
    nepoch;
    modelparamfile = "tempmodelparams_flux.jld2",
)
    #modelparamfile = "tempmodelparams_flux.jld2"
    for epoch = 1:nepoch
        loss_train = trainprocess!(θ, re, state, train_loader, lossfunction)
        #=
        for (x,y,num,totalnumatom) in train_loader
            xd = fmap(CuArray{Float64},x)
            yd = fmap(CuArray{Float64},y)
            L,dLdθ = Flux.withgradient(θ -> lossfunction(re(θ)(xd),yd),θ)
            Optimisers.update!(state, θ, dLdθ[1])
        end
        =#
        if epoch % 10 == 1 || epoch == nepoch
            loss, rmse, sse = testprocess(θ, re, state, test_loader, lossfunction)
            #=
            loss = 0.0
            rmse = 0.0
            for  (x,y,num,totalnumatom) in test_loader
                xd = fmap(CuArray{Float64},x)
                yd = fmap(CuArray{Float64},y)
                y_pred = re(θ)(xd) |> Flux.cpu_device()
                loss_t =lossfunction(y_pred,y) 
                loss += loss_t
                rmse += sqrt(loss_t) / (totalnumatom * train_loader.data.E_scale)
            end
            loss = loss / length(test_loader)
            =#
            θvalue = θ |> Flux.cpu_device()
            @save modelparamfile θvalue
            rmse = sqrt(sse) / test_loader.data.E_scale
            #println("Epoch: $epoch \t Loss: $loss rmse: $(rmse/length(test_loader)) [eV/atom] training mse: $loss_train")
            println(
                "Epoch: $epoch \t Loss: $loss rmse: $(rmse)) [eV/atom] training mse: $loss_train",
            )

        end
    end

end
export training!

function training!(θ, re, trainloader, testloader, lossfunction, nepoch)
    state = Optimisers.setup(Optimisers.AdamW(), θ)
    Ss = [get_S(istate, numspin) for istate = 1:(2^numspin)]
    for epoch = 1:nepoch
        loss = 0.0
        for (x, y, num, totalnumatom) in trainloader
            xd = fmap(CuArray{Float64}, x)
            yd = fmap(CuArray{Float64}, y)
            L, dLdθ = Flux.withgradient(θ -> lossfunction(re(θ)(xd), yd), θ)
            Optimisers.update!(state, θ, dLdθ[1])
            loss += L

        end
    end


    for it = 1:nt
        E, dEdθ = Flux.withgradient(θ -> calc_energy_full(H, re(θ), Ss), θ)
        Optimisers.update!(state, θ, dEdθ[1])
        println("$it energy: $E")
    end
    return θ, re
end

function train_batch!(
    x_train,
    y_train,
    model,
    loss,
    opt_state,
    x_test,
    y_test,
    nepoch,
    batchsize,
)
    numtestdata = size(y_test)[2]
    train_loader =
        Flux.DataLoader((x_train, y_train), batchsize = batchsize, shuffle = true)
    for it = 1:nepoch
        for (x, y) in train_loader
            grads = Flux.gradient(m -> loss(m(x), y), model)[1]
            Flux.update!(opt_state, model, grads)
        end

        if it % 10 == 0
            lossvalue = loss(model(x_test), y_test) / numtestdata
            println("$it-th testloss: $lossvalue")
        end
    end
end
