# Import modules to all processes
@everywhere using Pkg
@everywhere Pkg.activate("../..")
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using LinearAlgebra
# Import EKP modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
@everywhere include(joinpath(@__DIR__, "../../src/helper_funcs.jl"))
using JLD2
using NPZ

#########
#########  Define the parameters and their priors
#########

# Define the parameters that we want to learn
param_names = ["stochastic_noise"]
n_param = length(param_names)

# Prior information: Define transform to unconstrained gaussian space
constraints = [
    [bounded(0.5, 2.0)],
    ]
# All vars are standard Gaussians in unconstrained space
prior_dist = [Parameterized(Normal(0.0, 1.0))
                for x in range(1, n_param, length=n_param) ]
priors = ParameterDistribution(prior_dist, constraints, param_names)

# Define observation window (s)
ti = [4.0] * 3600
tf = [6.0] * 3600
# Define variables considered in the loss function
y_names = Array{String, 1}[]
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"])

# Define preconditioning and regularization of inverse problem
perform_PCA = true # Performs PCA on data

# Define name of PyCLES simulation to learn from
sim_names = ["Bomex"]
sim_suffix = [".may18"]
scm_sim_names = ["StochasticBomex"]  # corresponding scm dir

# Init arrays
yt = zeros(0)
yt_var_list = []
P_pca_list = []
pool_var_list = []
for (i, sim_name) in enumerate(sim_names)
    # "/Users/haakon/Documents/CliMA/SEDMF/LES_data/Output."
    les_dir = string("/groups/esm/ilopezgo/Output.", sim_name, sim_suffix[i])
    # Get SCM vertical levels for interpolation
    z_scm = get_profile(string("Output.", scm_sim_names[i], ".00000"), ["z_half"])
    # Get (interpolated and pool-normalized) observations, get pool variance vector
    yt_, yt_var_, pool_var = obs_LES(y_names[i], les_dir, ti[i], tf[i], z_scm = z_scm)
    push!(pool_var_list, pool_var)
    if perform_PCA
        yt_pca, yt_var_pca, P_pca = obs_PCA(yt_, yt_var_)
        append!(yt, yt_pca)
        push!(yt_var_list, yt_var_pca)
        push!(P_pca_list, P_pca)
    else
        append!(yt, yt_)
        push!(yt_var_list, yt_var_)
        global P_pca_list = nothing
    end
    # Save full dimensionality (normalized) output for error computation
end
d = length(yt) # Length of data array

# Construct global observational covariance matrix, TSVD
yt_var = zeros(d, d)
vars_num = 1
for (k,config_cov) in enumerate(yt_var_list)
    vars = length(config_cov[1,:])
    yt_var[vars_num:vars_num+vars-1, vars_num:vars_num+vars-1] = config_cov
    global vars_num = vars_num+vars
end

n_samples = 1
samples = zeros(n_samples, length(yt))
samples[1,:] = yt
Γy = yt_var

#########
#########  Run ensemble of simulations
#########

algo = Inversion() # Sampler(vcat(get_mean(priors)...), get_cov(priors))
N_ens = 1  # number of ensemble members
println("NUMBER OF ENSEMBLE MEMBERS: ", N_ens)

initial_params = construct_initial_ensemble(priors, N_ens, rng_seed=100)  # rand(1:1000)
ekobj = EnsembleKalmanProcess(initial_params, yt, Γy, algo )
scm_dir = "SCAMPy/"  # "/Users/haakon/Documents/CliMA/SCAMPy/"
scampy_handler = "call_stochasticBOMEX.sh"

# Define caller function
@everywhere g_(x::Array{Float64,1}) = run_SCAMPy(x, $param_names,
   $y_names, $scm_dir, $ti, $tf, P_pca_list = $P_pca_list,
   norm_var_list = $pool_var_list, scampy_handler = $scampy_handler)

# Create output dir
outdir_path = string("results_ensemble", "_p", n_param,"_e", N_ens, "_i", d)
println("Name of outdir path for this EKP, ", outdir_path)
command = `mkdir $outdir_path`
try
    run(command)
catch e
    println("Output directory already exists. Output may be overwritten.")
end

# Ensemble iteration
g_ens = zeros(N_ens, d)

# Note that the parameters are transformed when used as input to SCAMPy
params_cons_i = deepcopy(
    transform_unconstrained_to_constrained(
        priors, get_u_final(ekobj)
    )
)
params = [row[:] for row in eachrow(params_cons_i')]
@everywhere params = $params
## Run one ensemble forward map (in parallel)
array_of_tuples = pmap(g_, params) # Outer dim is params iterator
##
(g_ens_arr, g_ens_arr_pca) = ntuple(l->getindex.(array_of_tuples,l),2) # Outer dim is G̃, G 
i=1; println(string("\n\nEKP evaluation ",i," finished. \n"))
for j in 1:N_ens
    g_ens[j, :] = g_ens_arr_pca[j]
end

# update ensamble and compute error (should be unecessary)
update_ensemble!(ekobj, Array(g_ens'), deterministic_forward_map=false)
println("\nEnsemble updated. Saving results to file...\n")

# Convert to arrays
phi_params = Array{Array{Float64,2},1}(transform_unconstrained_to_constrained(priors, get_u(ekobj)))
phi_params_arr = zeros(i+1, n_param, N_ens)
for (k,elem) in enumerate(phi_params)
    phi_params_arr[k,:,:] = elem
end

# Save EKP information to JLD2 file
save(
    string(outdir_path,"/ekp.jld2"),
    "ekp_u", transform_unconstrained_to_constrained(priors, get_u(ekobj)),
    "ekp_g", get_g(ekobj),
    "truth_mean", ekobj.obs_mean,
    "truth_cov", ekobj.obs_noise_cov,
    "ekp_err", ekobj.err,
    "P_pca", P_pca_list,
    "pool_var", pool_var_list,
    "phi_params", phi_params_arr,
)

# Or you can also save information to numpy files with NPZ
npzwrite(string(outdir_path,"/y_mean.npy"), ekobj.obs_mean)
npzwrite(string(outdir_path,"/Gamma_y.npy"), ekobj.obs_noise_cov)
npzwrite(string(outdir_path,"/ekp_err.npy"), ekobj.err)
npzwrite(string(outdir_path,"/phi_params.npy"), phi_params_arr)
for (l, P_pca) in enumerate(P_pca_list)
    npzwrite(string(outdir_path,"/P_pca_",sim_names[l],".npy"), P_pca)
    npzwrite(string(outdir_path,"/pool_var_",sim_names[l],".npy"), pool_var_list[l])
end
