import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from netCDF4 import Dataset
import os

def read_scm_data(scm_data_path):
    """
    Read data from netcdf file into a dictionary that can be used for plots
    Input:
    scm_data_path  - path to scampy netcdf dataset with simulation results
    """
    scm_data = Dataset(scm_data_path, 'r')
    
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "buoyancy_mean", "b_mix","u_mean", "v_mean", "tke_mean",\
                 "updraft_buoyancy", "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql", "updraft_thetal",\
                 "env_qr", "updraft_qr", "env_RH", "updraft_RH", "updraft_w", "env_w", "env_thetal",\
                 "massflux_h", "diffusive_flux_h", "total_flux_h", "diffusive_flux_u", "diffusive_flux_v",\
                 "massflux_qt","diffusive_flux_qt","total_flux_qt","turbulent_entrainment",\
                 "eddy_viscosity", "eddy_diffusivity", "mixing_length", "mixing_length_ratio",\
                 "entrainment_sc", "detrainment_sc", "massflux", "nh_pressure", "nh_pressure_b", "nh_pressure_adv", "nh_pressure_drag", "eddy_diffusivity",\
                 "Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov",\
                 "Hvar_dissipation", "QTvar_dissipation", "HQTcov_dissipation",\
                 "Hvar_entr_gain", "QTvar_entr_gain", "HQTcov_entr_gain",\
                 "Hvar_detr_loss", "QTvar_detr_loss", "HQTcov_detr_loss",\
                 "Hvar_shear", "QTvar_shear", "HQTcov_shear", "H_third_m", "QT_third_m", "W_third_m",\
                 "Hvar_rain", "QTvar_rain", "HQTcov_rain","tke_entr_gain","tke_detr_loss",\
                 "tke_advection","tke_buoy","tke_dissipation","tke_pressure","tke_transport","tke_shear"\
                ]

    data = {"z_half" : np.divide(np.array(scm_data["profiles/z_half"][:]),1000.0),\
            "t" : np.divide(np.array(scm_data["profiles/t"][:]),3600.0),\
            "rho_half": np.array(scm_data["reference/rho0_half"][:])}

    for var in variables:
        data[var] = []
        if var=="QT_third_m":
            data[var] = np.transpose(np.array(scm_data[f"profiles/{var}"][:, :]))*1e9  #g^3/kg^3
        elif ("qt" in var or "ql" in var or "qr" in var):
            try:
                data[var] = np.transpose(np.array(scm_data[f"profiles/{var}"][:, :])) * 1000  #g/kg
            except:
                data[var] = np.transpose(np.array(scm_data["profiles/w_mean" ][:, :])) * 0  #g/kg
        else:
            data[var] = np.transpose(np.array(scm_data[f"profiles/{var}"][:, :]))
            
    return data

def read_les_data(les_data):
    """
    Read data from netcdf file into a dictionary that can be used for plots
    Input:
    les_data - pycles netcdf dataset with specific fileds taken from LES stats file
    """
    variables = ["temperature_mean", "thetali_mean", "qt_mean", "ql_mean", "buoyancy_mean",\
                 "u_mean", "v_mean", "tke_mean","v_translational_mean", "u_translational_mean",\
                 "updraft_buoyancy", "updraft_fraction", "env_thetali", "updraft_thetali",\
                 "env_qt", "updraft_qt","env_RH", "updraft_RH", "env_ql", "updraft_ql",\
                 "diffusive_flux_u", "diffusive_flux_v","massflux","massflux_u", "massflux_v","total_flux_u", "total_flux_v",\
                 "qr_mean", "env_qr", "updraft_qr", "updraft_w", "env_w",  "env_buoyancy", "updraft_ddz_p_alpha",\
                 "thetali_mean2", "qt_mean2", "env_thetali2", "env_qt2", "env_qt_thetali",\
                 "tke_prod_A" ,"tke_prod_B" ,"tke_prod_D" ,"tke_prod_P" ,"tke_prod_T" ,"tke_prod_S",\
                 "Hvar_mean" ,"QTvar_mean" ,"env_Hvar" ,"env_QTvar" ,"env_HQTcov", "H_third_m", "QT_third_m", "W_third_m",\
                 "massflux_h" ,"massflux_qt" ,"total_flux_h" ,"total_flux_qt" ,"diffusive_flux_h" ,"diffusive_flux_qt"]

    data = {"z_half" : np.divide(np.array(les_data["z_half"][:]),1000.0),\
            "t" : np.divide(np.array(les_data["t"][:]),3600.0),\
            "rho": np.array(les_data["profiles/rho"][:]),\
            "p0": np.divide(np.array(les_data["profiles/p0"][:]),100.0)}

    for var in variables:
        data[var] = []
        if ("QT_third_m" in var ):
            data[var] = np.transpose(np.array(les_data["profiles/"  + var][:, :]))*1e9  #g^3/kg^3
        elif ("qt" in var or "ql" in var or "qr" in var):
            try:
                data[var] = np.transpose(np.array(les_data["profiles/"  + var][:, :])) * 1000  #g/kg
            except:
                data[var] = np.transpose(np.array(les_data["profiles/w_mean" ][:, :])) * 0  #g/kg
        else:
            data[var] = np.transpose(np.array(les_data["profiles/"  + var][:, :]))


    return data

def get_LES_data(rootpath=Path.cwd()):
    (rootpath / "plots/output/Bomex/all_variables/").mkdir(parents=True, exist_ok=True)
    les_data_path = rootpath / "les_data/Bomex.nc"
    if not les_data_path.is_file():
        (rootpath / "les_data").mkdir(parents=True, exist_ok=True)
        url_ = r"https://drive.google.com/uc?export=download&id=1h8LcxaoBVHqtxwoaynux_3PrUpi_8GfY"
        os.system(f"curl -sLo {les_data_path} '{url_}'")
    les_data = Dataset(les_data_path, 'r')
    return read_les_data(les_data)

def time_bounds(data, tmin, tmax):
    """Get index of time bounds for scm/les data """
    t0_ind = int(np.where(np.array(data["t"]) > tmin)[0][0])
    t1_ind = int(np.where(np.array(tmax<= data["t"]))[0][0])
    return t0_ind, t1_ind

def initialize_plot(nrows, ncols, labs, zmin, zmax):
    """ Initialize a new plotting frame """
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    for ax, lab in zip(axs.flatten(), labs):
        ax.grid(True)
        ax.set_xlabel(lab)
    
    # Set ylabel on leftmost plots
    for i in range(nrows):
        ax = axs[i,0] if (nrows > 1) else axs[0]
        ax.set_ylabel("z [km]")
        ax.set_ylim([zmin, zmax])
    
    return fig, axs

def finalize_plot(fig, axs, folder, title):
    """ Finalize and save plot """
    # fig.legend()
    axs.flatten()[-1].legend()
    fig.tight_layout()
    fig.savefig(Path(folder) / title)
    fig.clf()

def make_plot(
        axs,
        scm_data_files, les_data, 
        tmin, tmax, 
        scm_vars, les_vars,
        scm_label,
        color_mean="lightblue", color_realizations="darkblue",
    ):
    """ Plot profiles from stochastic SCAMPy simulations
    
    Parameters:
    -----------
    scm_data_files                  :: scm stats files
    les_data                        :: les stats file
    tmin, tmax                      :: lower and upper bound for time mean
    scm_vars, les_vars              :: lists of variable identifiers for scm and les data
    scm_label                       :: label name for scm mean line
    color_mean, color_realizations  :: line color for mean line and individual realizations
    """
    
    # plot LES data
    t0_les, t1_les = time_bounds(les_data, tmin, tmax)
    
    # Plot LES data
    for ax, var in zip(axs.flatten(), les_vars):
        ax.plot(
            np.nanmean(les_data[var][:, t0_les:t1_les], axis=1),
            les_data["z_half"], '-', color='gray', label='les', lw=3,
        )

    # Process and plot SCM data
    scm_z_half = scm_data_files[0]["z_half"]
    scm_processed = {}
    for scm_data in scm_data_files:
        t0_scm, t1_scm = time_bounds(scm_data, tmin, tmax)

        for ax, var in zip(axs.flatten(), scm_vars):
            mean_var = np.nanmean(scm_data[var][:, t0_scm:t1_scm], axis=1)
            # Stack every realization of a variable to common array (for mean & std later) 
            if var in scm_processed:
                scm_processed[var] = np.vstack([scm_processed[var], mean_var])
            else:
                scm_processed[var] = mean_var
        
            # Add each realization to the plot
            ax.plot(
                mean_var, scm_z_half, "-", 
                color=color_realizations, lw=1, alpha=0.2,
            )

    # Plot mean and std
    for ax, var in zip(axs.flatten(), scm_vars):
        scm_var = scm_processed[var]
        mean_var = np.nanmean(scm_var, axis=0)
        std_var = np.std(scm_var, axis=0)
        ax.plot(  # plot mean value
            mean_var, scm_z_half, "-", 
            color=color_mean, label=scm_label, lw=3,
        )
        ax.fill_betweenx(  # plot standard deviation
            scm_z_half, mean_var - std_var, mean_var + std_var,
            color=color_mean, alpha=0.4, 
        )


### GENERATE PLOTS
folder = Path.cwd() / "results_ensemble_p1_e20_i7"

# get all files in folder on the form `Stats.<CaseName>.nc`
files = Path(folder).glob("Stats.*.nc")

# Read all scm datafiles
scm_datasets = [read_scm_data(file) for file in files]

# get LES data
les_data = get_LES_data(folder)

# plotting args
t0 = 4
t1 = 6
zmin = 0.0
zmax = 2.2
outfolder = folder / "plots/output/"

# Mean variables and flux variables
mean_flux_scm_vars = mean_flux_les_vars = [
    "qt_mean",          "ql_mean",          # qt_mean??
    "total_flux_h",     "total_flux_qt",
]
mean_flux_labs = [
    "mean qt [g/kg]",                                                   "mean ql [g/kg]",
    r'$ \langle w^* \theta_l^* \rangle  \; [\mathrm{kg K /m^2s}]$',     r'$ \langle w^* q_t^* \rangle  \; [\mathrm{g /m^2s}]$',
]
nrows = ncols = 2
title="StochasticBomex_means_fluxes.pdf"
scm_label = "scm"
fig, axs = initialize_plot(nrows, ncols, mean_flux_labs, zmin, zmax)
make_plot(
    axs, scm_datasets, les_data, t0, t1, mean_flux_scm_vars, mean_flux_les_vars,
    scm_label, color_mean="royalblue", color_realizations="darkblue",
)
finalize_plot(fig, axs, outfolder, title)

# Variance and covariance terms (+ TKE)
covar_scm_vars = [
    "tke_mean",     "HQTcov_mean",
    "Hvar_mean",    "QTvar_mean", 
]
covar_les_vars = [
    "tke_mean",     "env_HQTcov",
    "env_Hvar",     "env_QTvar", 
]
covar_labs = [
    r'$TKE [\mathrm{m^2/s^2}]$',    "HQTcov",
    "Hvar",                         "QTvar",
]
nrows = ncols = 2
title="StochasticBomex_var_covar.pdf"
fig, axs = initialize_plot(nrows, ncols, covar_labs, zmin, zmax)
make_plot(
    axs, scm_datasets, les_data, t0, t1, covar_scm_vars, covar_les_vars,
    scm_label, color_mean="royalblue", color_realizations="darkblue",
    )
finalize_plot(fig, axs, outfolder, title)

# Third-order moments
third_scm_vars = third_scm_vars = [
    "H_third_m",    "QT_third_m",
]
third_labs = [
    r'$ \langle \theta_l^*\theta_l^*\theta_l^* \rangle [K^3] $',    r'$ \langle q_t^*q_t^*q_t^* \rangle [g^3/kg^3] $',
]
nrows = 1 
ncols = 2
title="StochasticBomex_third.pdf"
fig, axs = initialize_plot(nrows, ncols, third_labs, zmin, zmax)
make_plot(
    axs, scm_datasets, les_data, t0, t1, third_scm_vars, third_scm_vars, 
    scm_label, color_mean="royalblue", color_realizations="darkblue",
    )
finalize_plot(fig, axs, outfolder, title)
