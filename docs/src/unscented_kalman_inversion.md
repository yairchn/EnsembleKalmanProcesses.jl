# Unscented Kalman Inversion

## Algorithm
One of the ensemble Kalman processes implemented in `EnsembleKalmanProcesses.jl` is the unscented Kalman inversion ([Huang et al, 2021](https://arxiv.org/abs/2102.01580)). The unscented Kalman inversion (UKI) is a derivative-free ensemble optimization method that seeks to find the optimal parameters $\theta \in \mathbb{R}^p$ in the inverse problem
```math
 y = \mathcal{G}(\theta) + \eta
```
where $\mathcal{G}$ denotes the forward map, $y \in \mathbb{R}^d$ is the vector of observations and $\eta \sim \mathcal{N}(0, \Gamma_y)$ is additive Gaussian observational noise. Note that $p$ is the size of the parameter vector $\theta$ and $d$ is taken to be the size of the observation vector $y$. 
The UKI updates both the mean $m_n$ and covariance $C_n$ estimations of the parameter vector $\theta$ as following

* Prediction step :

```math
    $$\begin{align*}
    \hat{m}_{n+1} = & r+\alpha(m_n-r)\\
    \hat{C}_{n+1} = & \alpha^2 C_{n} + \Sigma_{\omega}
    \end{align*}$$
```  
* Generate sigma points :
```math    
    $$\begin{align*}
    &\hat{\theta}_{n+1}^0 = \hat{m}_{n+1} \\
    &\hat{\theta}_{n+1}^j = \hat{m}_{n+1} + c_j [\sqrt{\hat{C}_{n+1}}]_j \quad (1\leq j\leq N_\theta)\\ 
    &\hat{\theta}_{n+1}^{j+N_\theta} = \hat{m}_{n+1} - c_j [\sqrt{\hat{C}_{n+1}}]_j\quad (1\leq j\leq N_\theta)
    \end{align*}$$
```     
*  Analysis step :
    
```math
   \begin{align*}
        &\hat{y}^j_{n+1} = \mathcal{G}(\hat{\theta}^j_{n+1}) \qquad \hat{y}_{n+1} = \hat{y}^0_{n+1}\\
         &\hat{C}^{\theta p}_{n+1} = \sum_{j=1}^{2N_\theta}W_j^{c}
        (\hat{\theta}^j_{n+1} - \hat{m}_{n+1} )(\hat{y}^j_{n+1} - \hat{y}_{n+1})^T \\
        &\hat{C}^{pp}_{n+1} = \sum_{j=1}^{2N_\theta}W_j^{c}
        (\hat{y}^j_{n+1} - \hat{y}_{n+1} )(\hat{y}^j_{n+1} - \hat{y}_{n+1})^T + \Sigma_{\nu}\\
        &m_{n+1} = \hat{m}_{n+1} + \hat{C}^{\theta p}_{n+1}(\hat{C}^{pp}_{n+1})^{-1}(y - \hat{y}_{n+1})\\
        &C_{n+1} = \hat{C}_{n+1} - \hat{C}^{\theta p}_{n+1}(\hat{C}^{pp}_{n+1})^{-1}{\hat{C}^{\theta p}_{n+1}}{}^{T}\\
    \end{align*}
```

The unscented transformation parameters are
```math
    \begin{align*}
    &c_j = \sqrt{N_\theta +\lambda} \qquad W_j^{c} = \frac{1}{2(N_\theta+\lambda)}~(j=1,\cdots,2N_{\theta}).\\
    &\lambda = a^2 (N_\theta + \kappa) - N_\theta \quad a=\min\{\sqrt{\frac{4}{N_\theta + \kappa}},  1\}\quad  \kappa = 0\\
    \end{align*}
```
And $[\sqrt{C}]_j$ is the $j$th column of the Cholesky factor of $C$. 
    
    

For typical applications, a near-optimal solution $\theta$ can be found after as few as 10 iterations of the algorithm, and no covariance inflation or early stopping is needed. The ensemble will not collapse, and therefore, the covariance estimation represents the uncertainty of the problem.


## Free parameters
The free parameters in the unscented Kalman inversion are

$$\alpha, r, \Sigma_{\nu}, \Sigma_{\omega}$$

They are chosen based on theorems developed in [Huang et al, 2021](https://arxiv.org/abs/2102.01580)

* $r$ is generally set to be the prior mean

* $\alpha \in (0,1]$ is a regularization parameter, which is used to overcome ill-posedness and overfitting. A practical guide is 

    * When the observation noise is negligible, and there are more observations than parameters (identifiable inverse problem) $\alpha = 1$ (no regularization)
    * Otherwise $\alpha < 1$. The smaller $\alpha$ is, the closer UKI will converge to the prior mean.
    
* $\Sigma_{\nu}$ is the artificial observation errror covariance. We choose $\Sigma_{\nu} = 2 \Sigma_{\eta}$, which makes the inverse problem consistent. 

* $\Sigma_{\omega}$ is the artificial evolution errror covariance. We choose $\Sigma_{\omega} = (2 - \alpha^2)\Lambda$

* When there are more observations than parameters (identifiable inverse problem), $\Lambda = C_n$, which is updated as the estimated covariance $C_n$ in the $n$-thevery iteration . This guarantees the converged covariance matrix is a good approximation to the posterior covariance matrix with an uninformative prior.
    
* Otherwise $\Lambda = C_0$, this allows that the converged covariance matrix is a weighted average between the posterior covariance matrix with an uninformative prior and $C_0$.

Therefore, the user only need to change the $\alpha$, and the freqency to update the $\Lambda$.


## Implementation

### Initialization
An unscented Kalman inversion object can be created using the `EnsembleKalmanProcess` constructor by specifying the `Unscented()` process type.

Creating an ensemble Kalman inversion object requires as arguments:
 1. The mean value of the observed outputs, a vector of size `[d]`;
 2. The covariance of the observational noise, a matrix of size `[d × d]`;
 3. The `Unscented()` process type.

The initialization of the `Unscented()` process requires prior mean and prior covariance, and the the size of the observation `d`. And user defined hyperparameters 
`α_reg` and `update_freq`.
```julia
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage


# need to choose regularization factor α ∈ (0,1],  
# when you have enough observation data α=1: no regularization
α_reg =  1.0
# update_freq 1 : approximate posterior covariance matrix with an uninformative prior
#             0 : weighted average between posterior covariance matrix with an uninformative prior and prior
update_freq = 1

process = Unscented(prior_mean, prior_cov, length(truth_sample), α_reg, update_freq)
ukiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)

```

Note that no information about the forward map is necessary to initialize the Inversion process. The only forward map information required by the inversion process consists of model evaluations at the ensemble elements, necessary to update the ensemble.

### Updating the Ensemble

Once the unscented Kalman inversion object `UKIobj` has been initialized, any number of updates can be performed using the inversion algorithm.

A call to the inversion algorithm can be performed with the `update_ensemble!` function. This function takes as arguments the `UKIobj` and the evaluations of the forward map at each element of the current ensemble. The `update_ensemble!` function then stores the new updated ensemble and the inputted forward map evaluations in `UKIobj`.

A typical use of the `update_ensemble!` function given the ensemble Kalman inversion object `UKIobj` and the forward map `G` is
```julia
N_iter = 20 # Number of steps of the algorithm


for i in 1:N_iter

    params_i = get_u_final(ukiobj); dims=1)

    # define black box parameter to observation map, 
    # with certain parameter transformation related to imposing some constraints
    # i.e. θ -> e^θ  -> G(e^θ) = y
    g_ens = g_ens = GModel.run_G_ensemble(params_i, lorenz_settings_G)
    # analysis step 
    EnsembleKalmanProcessModule.update_ensemble!(ukiobj, g_ens) 
end
```