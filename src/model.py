import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import numpyro
import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import HalfCauchy
from numpyro.distributions import InverseGamma
from numpyro.distributions import Normal
import numpy as np


print("---------------------------- Model Define -------------------------")
############################## Horse-shoe dist ################################
def _sample_reg_horseshoe(tau: float, c_sq: float, shape: tuple[int, ...],name = "betaH"):

    lamb = numpyro.sample(name+"_λ", HalfCauchy(1.0), sample_shape=shape)
    lamb_squiggle = jnp.sqrt(c_sq) * lamb / jnp.sqrt(c_sq + tau**2 * lamb**2)
    betaH = numpyro.sample(
        name,
        Normal(jnp.zeros_like(lamb_squiggle), jnp.sqrt(lamb_squiggle**2 * tau**2)),
    )
    return betaH
############################## Horse-shoe Model ################################
def MultiTargetMultiEquation_HSModel(xx, yy):
    n_targets = yy.shape[1]

    n_IDs = xx.shape[0]
    n_features = xx.shape[1]
    n_obs = xx.shape[2]

    slab_shape_nu = 4
    slab_shape_s = 2
    noise_hyper_lambda = 1
    sparsity_coef_tau0 = 0.1
    print(f"n_features = {n_features}")
    # sample the horseshoe hyperparameters.
    τ_μ = numpyro.sample("τ_μ", HalfCauchy(sparsity_coef_tau0),sample_shape =(n_IDs,n_features,n_targets) )
    c_sq_μ = numpyro.sample("c_sq_μ",InverseGamma(slab_shape_nu / 2, slab_shape_nu / 2 * slab_shape_s**2),sample_shape =(n_IDs,n_features,n_targets))
    μ_coef = _sample_reg_horseshoe(τ_μ, c_sq_μ,( n_IDs,n_features,n_targets),"μ_coef")
    print(f"μ_coef.shape = {μ_coef.shape}")
    σ_coef = numpyro.sample("σ_coef", dist.HalfNormal(10.),sample_shape =(n_IDs,n_features,n_targets))


    with numpyro.plate("plate_targets", n_targets):
      with numpyro.plate("plate_features", n_features):
        with numpyro.plate("plate_Indv", n_IDs):
          coef = numpyro.sample("coef", dist.Normal(μ_coef, σ_coef))

    y_est = jnp.einsum('ifo,ift->ito', xx, coef)
    # for each target we should consider different noise
    noise = numpyro.sample("noise", dist.HalfNormal(1),sample_shape=(n_targets,))
    noise = jnp.expand_dims(jnp.expand_dims(noise, axis=1), axis=0)
    noise = jnp.broadcast_to(noise, (n_IDs,n_targets,n_obs))

    numpyro.sample("obs", dist.Normal(y_est, noise), obs=yy)


################################# Flat Bayesian Horseshoe Model ############################
# def pre_Flat(xx, yy):
#     n_ind, n_coef, n_obs = xx.shape
#     n_ind, n_eqs, n_obs =yy.shape
#     x_out = np.reshape(xx, (n_coef, n_ind * n_obs))
#     y_out = np.reshape(yy, (n_eqs, n_ind * n_obs))
#     return x_out, y_out
def pre_Flat(xx, yy):
    n_indv, n_coef, n_obs = xx.shape
    n_indv, n_eq, n_obs = yy.shape
    xx_out = []
    yy_out = []

    for coef_i in range(n_coef):
        # Take the data for the current coefficient across all individuals and observations
        xx_coef_data = xx[:, coef_i, :] # Shape (n_indv, n_obs)
        # Reshape to (n_indv * n_obs,) and append to xx_out
        xx_out.append(xx_coef_data.reshape(-1))

    for eq_i in range(n_eq):
        # Take the data for the current equation across all individuals and observations
        yy_eq_data = yy[:, eq_i, :] # Shape (n_indv, n_obs)
        # Reshape to (n_indv * n_obs,) and append to yy_out
        yy_out.append(yy_eq_data.reshape(-1))

    # Stack the lists to get arrays of shape (n_coef, n_indv * n_obs) and (n_eq, n_indv * n_obs)
    xx_out = np.stack(xx_out, axis=0)
    yy_out = np.stack(yy_out, axis=0)

    print(xx_out.shape)
    print(yy_out.shape)

    return xx_out, yy_out

def Flat_HSModel(xx, yy):
    n_targets = yy.shape[0]

    n_features = xx.shape[0]
    n_obs = xx.shape[1]

    slab_shape_nu = 4
    slab_shape_s = 2
    noise_hyper_lambda = 1
    sparsity_coef_tau0 = 0.1
    print(f"n_features = {n_features}")
    # sample the horseshoe hyperparameters.
    τ_μ = numpyro.sample("τ_μ", HalfCauchy(sparsity_coef_tau0))
    c_sq_μ = numpyro.sample("c_sq_μ",InverseGamma(slab_shape_nu / 2, slab_shape_nu / 2 * slab_shape_s**2))
    coef = _sample_reg_horseshoe(τ_μ, c_sq_μ,( n_features,n_targets),"coef")
    print(f"coef = {coef.shape}")
    print(f"xx = {xx.shape}")

    y_est = jnp.einsum('fo,ft->to', xx, coef)
    print(f"y_est = {y_est.shape}")
    # for each target we should consider different noise
    noise = numpyro.sample("noise", dist.HalfNormal(1),sample_shape=(n_targets,))
    noise = jnp.expand_dims(noise, axis=1)
    noise = jnp.broadcast_to(noise, (n_targets,n_obs))
    print(f"noise = {noise.shape}")

    numpyro.sample("obs", dist.Normal(y_est, noise), obs=yy)















    ################################### Normal Model ########################
    def MultiTargetMultiEquation_Normal(xx, yy):
        n_targets = yy.shape[1]

        n_IDs = xx.shape[0]
        n_features = xx.shape[1]
        n_obs = xx.shape[2]

        μ_coef = numpyro.sample("μ_coef", dist.Normal(0, 10.), sample_shape=(n_IDs, n_features, n_targets))
        σ_coef = numpyro.sample("σ_coef", dist.HalfNormal(10.), sample_shape=(n_IDs, n_features, n_targets))

        with numpyro.plate("plate_targets", n_targets):
            with numpyro.plate("plate_features", n_features):
                with numpyro.plate("plate_Indv", n_IDs):
                    coef = numpyro.sample("coef", dist.Normal(μ_coef, σ_coef))

        y_est = jnp.einsum('ifo,ift->ito', xx, coef)
        # for each target we should consider different noise
        noise = numpyro.sample("noise", dist.HalfNormal(1), sample_shape=(n_targets,))
        noise = jnp.expand_dims(jnp.expand_dims(noise, axis=1), axis=0)
        noise = jnp.broadcast_to(noise, (n_IDs, n_targets, n_obs))

        numpyro.sample("obs", dist.Normal(y_est, noise), obs=yy)

