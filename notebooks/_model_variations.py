import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from functools import partial
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,)
import click
import pickle as pkl
import time
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs) 

def basic_bernoulli_adjacency1(
        spectral_count):
    n_prey = 3_062
#    assert len(spectral_count.preyu) == n_prey
    shape = (n_prey, n_prey)
    A = jnp.ones(shape, dtype=bool)
    probs = A * 0.01 
    probs = probs * (jnp.eye(n_prey) == 0)
#    A = numpyro.sample('A', dist.Normal(probs))
    A = numpyro.sample('A', dist.RelaxedBernoulli(probs,
        temperature=0.5))
#    L = jnp.tril(A, k=-1)
#    A = A + A.T

def relaxed_bernoulli1(x):
    n_prey = 36
    event_shape = (n_prey, n_prey)
    A = jnp.ones(event_shape, dtype=bool)
    probs = A * 0.01 
    probs = probs * (jnp.eye(n_prey) == 0)
    A = numpyro.sample('A', dist.RelaxedBernoulli(
        temperature=0.5, probs=probs))
    L = jnp.tril(A, k=-1)
    A = L + L.T

def model(y):  # y has dimension N x d
#    d = y.shape[1]
#    N = y.shape[0]
    N = 4
    d = 3_062
    y = jnp.ones((4, d))
    # Vector of variances for each of the d variables
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    # Lower cholesky factor of a correlation matrix
    concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(d, concentration))
    # Lower cholesky factor of the covariance matrix
    sigma = jnp.sqrt(theta)
    # we can also use a faster formula `L_Omega = sigma[..., None] * L_omega`
    L_Omega = jnp.matmul(jnp.diag(sigma), L_omega)
    # Vector of expectations
    mu = jnp.zeros(d)
    with numpyro.plate("observations", N):
        obs = numpyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=y)
    return obs

def model(y):
    N = 4
    d = 3_062
    y = jnp.ones((N, d))
    K_corr = numpyro.sample('K', dist.LKJ(d, 1.5))
    mu = jnp.zeros(d)
    with numpyro.plate("observations", N):
        obs = numpyro.sample("obs", dist.MultivariateNormal(mu, precision_matrix=K), obs=y)
    return obs

def model(y):
    N = 4
    d = 30 #3_062
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    y = jnp.ones((N, d))
    U = numpyro.sample('U', dist.LKJ(d, 1.5))
    mu = jnp.zeros(d)
    obs = numpyro.sample("obs", dist.MultivariateNormal(
        mu, precision_matrix=U), obs=y)

def model(y):  # y has dimension N x d
    N = 4
    d = 10 
    y = jnp.ones((N, d))
    y1 = y[0, :].reshape((d, 1))
    y2 = y[1, :].reshape((d, 1))
    y3 = y[2, :].reshape((d, 1))
    y4 = y[3, :].reshape((d, 1))
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
    corr_mat = numpyro.sample("corr_mat", dist.LKJ(d, concentration))
    sigma = jnp.sqrt(theta)
    cov_mat = jnp.outer(theta, theta) * corr_mat
    cov_mat = jnp.tril(cov_mat) + jnp.tril(cov_mat, k=-1).T 
    mu = jnp.zeros(d)
    obs1 = numpyro.sample("obs1", dist.MultivariateNormal(mu, cov_matrix=cov_mat), obs=y1)
    obs2 = numpyro.sample("obs2", dist.MultivariateNormal(mu, cov_matrix=cov_mat), obs=y2)
    obs3 = numpyro.sample("obs3", dist.MultivariateNormal(mu, cov_matrix=cov_mat), obs=y3)
    obs4 = numpyro.sample("obs4", dist.MultivariateNormal(mu, cov_matrix=cov_mat), obs=y4)

def model(y): # y has dimension N x d
    d = y.shape[1]
    print(f"y: {y.shape}")
    N = y.shape[0]
    # Vector of variances for each of the d variables
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    print(f"theta: {theta.shape}")
    concentration = jnp.ones(1) # Implies a uniform distribution over
    prior = dist.LKJ(d, concentration)
    print(f"batch: {prior.batch_shape}")
    print(f"event {prior.event_shape}")
    #print(f"sample: {prior.sample_shape}")
    corr_mat = numpyro.sample("corr_mat", prior)
    assert jnp.all(corr_mat[0, :, :] == cor_mat[0, :, :])
    print(f"corr_mat: {corr_mat.shape}")
    print(f"corr_mat info {type(corr_mat)}")
    sigma = jnp.sqrt(theta)
    # we can also use a faster formula `cov_mat = jnp.outer(theta, theta) * corr_
    cov_mat = jnp.matmul(jnp.matmul(jnp.diag(sigma), corr_mat), jnp.diag(sigma))
    print(f"cov_mat: cov_mat.shape")
    #L = jnp.tril(cov_mat)
    #cov_mat = L + jnp.tril(L, k=-1).T
    # Vector of expectations
    mu = jnp.zeros(d)
    print(f"mu: {mu.shape}")
    with numpyro.plate("observations", N):
        obs = numpyro.sample("obs", dist.MultivariateNormal(mu, covariance_matrix=cov_mat), obs=y)
        return obs

#assert jnp.all(cov_mat == cov_mat.T), jnp.where(
#    cov_mat != cov_mat.T)
def eight_schools(J, sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', J):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

def cov_model1(dim):
    Sigma = numpyro.sample("s", dist.LKJCholesky(dim, 1.1))
    print(Sigma)
#    numpyro.sample("obs", dist.MultivariateNormal(jnp.arange(dim), scale_tril=Sigma),
#                   obs=jnp.arange(dim))
#    return Sigma

def cov_model2(dim):
    """
    Model works
    """
    mu = jnp.zeros(dim)
    sigma = jnp.eye(dim)
    numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, covariance_matrix=sigma))


def cov_model3(dim):
    """
    """
    #mu = jnp.zeros(dim)
    sigma = numpyro.sample("sigma", dist.LKJCholesky(dim, concentration=jnp.array(1.5, dtype=jnp.float64)))
    #numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, covariance_matrix=sigma))

def get_cov_model4_init_strategy(dim):
    values = {"sigma": jnp.eye(dim),
              "mvn": jnp.zeros(dim)}
    return partial(init_to_value, values=values)

def cov_model4(dim):
    """
    """
    mu = jnp.zeros(dim)
    sigma = numpyro.sample("sigma", dist.LKJCholesky(dim, concentration=jnp.array(1.5, dtype=jnp.float64)))
    #This is wierdly misspecified
    numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, covariance_matrix=sigma))

def cov_model5(dim):
    """
    Use model4 init strategy
    """
    mu = jnp.zeros(dim)
    sigma = numpyro.sample("sigma", dist.LKJCholesky(dim, concentration=1.5))
    #This is wierdly misspecified
    numpyro.sample("mvn", dist.MultivariateNormal(loc=mu, scale_tril=sigma), obs=jnp.zeros(dim))

def load(fpath):
    with open(fpath, 'rb') as f:
        dat = pkl.load(f)
    return dat
   
@click.command()
@click.option("--model-id", type=str, help="identifier is prepended to saved files")
@click.option("--rseed", type=int, default=0)
@click.option("--model-name", type=str)
@click.option("--model-data")
@click.option("--num-warmup", type=int)
@click.option("--num-samples", type=int)
@click.option("--include-potential-energy", is_flag=True, default=False)
@click.option("--progress-bar", is_flag=True, default=False)
def main(model_id, rseed, model_name,model_data, num_warmup, num_samples, include_potential_energy, 
    progress_bar):
    entered_main_time = time.time()
    rng_key = jax.random.PRNGKey(rseed)
    eprint(f"Model Name: {model_name}")
    if model_name == "cov_model5":
        model_data = int(model_data)
        model = cov_model5
        init_strategy = get_cov_model4_init_strategy(model_data)
    extra_fields = ()
    if include_potential_energy:
        extra_fields = extra_fields + ("potential_energy", )
    nuts = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples, progress_bar=progress_bar)
    mcmc.run(rng_key, model_data, extra_fields=extra_fields)
    finished_mcmc_time = time.time()
    elapsed_time = finished_mcmc_time - entered_main_time
    eprint(f"{num_samples} sampling steps complete")
    eprint(f"elapsed time: {elapsed_time}")
    savename = model_id + "_" + model_name + "_" + str(rseed) + ".pkl"
    fields = mcmc.get_extra_fields()
    samples = mcmc.get_samples()
    out = {"extra_fields": fields, "samples": samples, "elapsed_time": elapsed_time}
    with open(savename, "wb") as file:
        pkl.dump(out, file) 

if __name__ == "__main__":
    main()

#model_id, rseed, model_name,
#         model_data, num_warmup, num_samples, include_potential_energy
