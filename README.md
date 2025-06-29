# BPNetTrainer.jl


This repository aims to port Flux.jl implementation [BPNET.jl](https://github.com/cometscome/BPNET.jl) to Lux.jl

# Usage

1. Install Julia
1. Install Revise.jl

```
$ julia -e 'using Pkg; Pkg.add("Revise")'
```

## Running training BPNet with Flux.jl

```sh
$ julia --project playground/pluto/flux_model_training_example.jl
```

## Running training BPNet with Lux.jl

```sh
$ julia --project playground/pluto/lux_model_training_example.jl
```

## Running training BPNet with Optimization/LBFGS

```sh
julia --project playground/pluto/optimization_model_training_example.jl
```
