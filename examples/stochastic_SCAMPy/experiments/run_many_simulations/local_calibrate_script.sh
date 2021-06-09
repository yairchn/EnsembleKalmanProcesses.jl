julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

julia --project -p 1 ensemble_run.jl