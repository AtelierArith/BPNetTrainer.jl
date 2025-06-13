# DDPTrainingExamples.jl

## Setup

1. Install julia via `juliaup`
        ```sh
        $ julia --project -e 'using Pkg; Pkg.instantiate()'
1. Install `mpiexecjl` command:
        ```sh
        $ julia --project -e 'using MPI; MPI.install_mpiexecjl()'
        ```
1. Run the following command:
        ```sh
        $ ~/.julia/bin/mpiexecjl -np 2 julia -t auto --project ddp.jl
        ```
